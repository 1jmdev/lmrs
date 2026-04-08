use std::ffi::{CStr, CString};
use std::path::Path;
use std::ptr::NonNull;
use std::sync::Once;

use crate::error::LmrsError;
use crate::ffi;
use crate::message::Message;

static BACKEND_INIT: Once = Once::new();

#[derive(Clone, Copy, Debug)]
pub struct GenerationConfig {
    pub max_tokens: usize,
    pub context_size: u32,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_tokens: 128,
            context_size: 2048,
        }
    }
}

pub struct LlamaRuntime {
    model: NonNull<ffi::llama_model>,
    ctx: NonNull<ffi::llama_context>,
    vocab: *const ffi::llama_vocab,
    vocab_size: usize,
}

impl LlamaRuntime {
    pub fn load(model_path: impl AsRef<Path>) -> Result<Self, LmrsError> {
        Self::load_with_config(model_path, GenerationConfig::default())
    }

    pub fn load_with_config(
        model_path: impl AsRef<Path>,
        config: GenerationConfig,
    ) -> Result<Self, LmrsError> {
        unsafe extern "C" fn noop_log(
            _level: ffi::ggml_log_level,
            _text: *const std::os::raw::c_char,
            _user_data: *mut std::os::raw::c_void,
        ) {
        }

        unsafe {
            ffi::llama_log_set(Some(noop_log), std::ptr::null_mut());
        }

        BACKEND_INIT.call_once(|| unsafe {
            ffi::llama_backend_init();
        });

        let model_path = model_path.as_ref();
        let model_path = model_path.to_str().ok_or(LmrsError::ModelPathNotUtf8)?;
        let model_path = CString::new(model_path).map_err(|_| LmrsError::CStringNul)?;

        let mut model_params = unsafe { ffi::llama_model_default_params() };
        model_params.n_gpu_layers = 0;

        let model = unsafe { ffi::llama_model_load_from_file(model_path.as_ptr(), model_params) };
        let model = NonNull::new(model).ok_or(LmrsError::ModelLoadFailed)?;

        let vocab = unsafe { ffi::llama_model_get_vocab(model.as_ptr()) };
        let vocab_size = unsafe { ffi::llama_vocab_n_tokens(vocab) as usize };

        let mut ctx_params = unsafe { ffi::llama_context_default_params() };
        ctx_params.n_ctx = config.context_size;
        ctx_params.n_batch = config.context_size;
        ctx_params.n_seq_max = 1;
        ctx_params.n_threads = thread_count();
        ctx_params.n_threads_batch = thread_count();
        ctx_params.flash_attn_type = ffi::llama_flash_attn_type_LLAMA_FLASH_ATTN_TYPE_DISABLED;
        ctx_params.offload_kqv = false;

        let ctx = unsafe { ffi::llama_init_from_model(model.as_ptr(), ctx_params) };
        let ctx = match NonNull::new(ctx) {
            Some(ctx) => ctx,
            None => {
                unsafe { ffi::llama_model_free(model.as_ptr()) };
                return Err(LmrsError::ContextInitFailed);
            }
        };

        Ok(Self {
            model,
            ctx,
            vocab,
            vocab_size,
        })
    }

    pub fn generate(
        &self,
        messages: &[Message],
        config: GenerationConfig,
    ) -> Result<String, LmrsError> {
        let prompt = self.apply_chat_template(messages)?;
        let mut prompt_tokens = self.tokenize(&prompt)?;
        if prompt_tokens.is_empty() {
            return Ok(String::new());
        }

        self.decode(&mut prompt_tokens)?;

        let mut generated = Vec::with_capacity(config.max_tokens);
        let mut output_bytes = Vec::new();

        for _ in 0..config.max_tokens {
            let token = self.sample_greedy()?;

            if unsafe { ffi::llama_vocab_is_eog(self.vocab, token) } {
                break;
            }

            output_bytes.extend(self.token_to_piece_bytes(token)?);
            generated.push(token);

            let mut next = [token];
            self.decode(&mut next)?;
        }

        Ok(String::from_utf8_lossy(&output_bytes).trim().to_string())
    }

    fn apply_chat_template(&self, messages: &[Message]) -> Result<String, LmrsError> {
        let role_strings = messages
            .iter()
            .map(|message| CString::new(message.role).map_err(|_| LmrsError::CStringNul))
            .collect::<Result<Vec<_>, _>>()?;
        let content_strings = messages
            .iter()
            .map(|message| {
                CString::new(message.content.replace('\0', "")).map_err(|_| LmrsError::CStringNul)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let chat_messages = role_strings
            .iter()
            .zip(content_strings.iter())
            .map(|(role, content)| ffi::llama_chat_message {
                role: role.as_ptr(),
                content: content.as_ptr(),
            })
            .collect::<Vec<_>>();

        let template = unsafe {
            let ptr = ffi::llama_model_chat_template(self.model.as_ptr(), std::ptr::null());
            if ptr.is_null() {
                CString::new("llama3").map_err(|_| LmrsError::CStringNul)?
            } else {
                CString::new(CStr::from_ptr(ptr).to_bytes()).map_err(|_| LmrsError::CStringNul)?
            }
        };

        let estimate = messages
            .iter()
            .map(|message| message.role.len() + message.content.len() + 64)
            .sum::<usize>()
            .max(512);
        let mut buffer = vec![0u8; estimate];

        let written = unsafe {
            ffi::llama_chat_apply_template(
                template.as_ptr(),
                chat_messages.as_ptr(),
                chat_messages.len(),
                true,
                buffer.as_mut_ptr().cast(),
                buffer.len() as i32,
            )
        };

        if written < 0 {
            return Err(LmrsError::TemplateApplyFailed);
        }

        if written as usize > buffer.len() {
            buffer.resize(written as usize, 0);
            let rewritten = unsafe {
                ffi::llama_chat_apply_template(
                    template.as_ptr(),
                    chat_messages.as_ptr(),
                    chat_messages.len(),
                    true,
                    buffer.as_mut_ptr().cast(),
                    buffer.len() as i32,
                )
            };
            if rewritten < 0 {
                return Err(LmrsError::TemplateApplyFailed);
            }
            return Ok(String::from_utf8_lossy(&buffer[..rewritten as usize]).into_owned());
        }

        Ok(String::from_utf8_lossy(&buffer[..written as usize]).into_owned())
    }

    fn tokenize(&self, text: &str) -> Result<Vec<ffi::llama_token>, LmrsError> {
        let mut tokens = vec![0; text.len() + 8];
        let mut count = unsafe {
            ffi::llama_tokenize(
                self.vocab,
                text.as_ptr().cast(),
                text.len() as i32,
                tokens.as_mut_ptr(),
                tokens.len() as i32,
                true,
                false,
            )
        };

        if count < 0 {
            tokens.resize((-count) as usize, 0);
            count = unsafe {
                ffi::llama_tokenize(
                    self.vocab,
                    text.as_ptr().cast(),
                    text.len() as i32,
                    tokens.as_mut_ptr(),
                    tokens.len() as i32,
                    true,
                    false,
                )
            };
        }

        if count < 0 {
            return Err(LmrsError::TokenizeFailed(count));
        }

        tokens.truncate(count as usize);
        Ok(tokens)
    }

    fn decode(&self, tokens: &mut [ffi::llama_token]) -> Result<(), LmrsError> {
        let batch = unsafe { ffi::llama_batch_get_one(tokens.as_mut_ptr(), tokens.len() as i32) };
        let result = unsafe { ffi::llama_decode(self.ctx.as_ptr(), batch) };
        if result == 0 {
            Ok(())
        } else {
            Err(LmrsError::DecodeFailed(result))
        }
    }

    fn sample_greedy(&self) -> Result<ffi::llama_token, LmrsError> {
        let logits = unsafe { ffi::llama_get_logits_ith(self.ctx.as_ptr(), -1) };
        let logits = NonNull::new(logits).ok_or(LmrsError::NoLogits)?;
        let logits = unsafe { std::slice::from_raw_parts(logits.as_ptr(), self.vocab_size) };

        let token = logits
            .iter()
            .enumerate()
            .max_by(|(_, left), (_, right)| left.total_cmp(right))
            .map(|(index, _)| index as ffi::llama_token)
            .ok_or(LmrsError::NoLogits)?;

        Ok(token)
    }

    fn token_to_piece_bytes(&self, token: ffi::llama_token) -> Result<Vec<u8>, LmrsError> {
        let mut buffer = vec![0u8; 64];

        loop {
            let written = unsafe {
                ffi::llama_token_to_piece(
                    self.vocab,
                    token,
                    buffer.as_mut_ptr().cast(),
                    buffer.len() as i32,
                    0,
                    true,
                )
            };

            if written >= 0 && written as usize <= buffer.len() {
                buffer.truncate(written as usize);
                return Ok(buffer);
            }

            if written < 0 {
                buffer.resize((-written) as usize, 0);
                let retry = unsafe {
                    ffi::llama_token_to_piece(
                        self.vocab,
                        token,
                        buffer.as_mut_ptr().cast(),
                        buffer.len() as i32,
                        0,
                        true,
                    )
                };
                if retry < 0 {
                    return Err(LmrsError::TokenToPieceFailed(retry));
                }
                buffer.truncate(retry as usize);
                return Ok(buffer);
            }

            buffer.resize(written as usize, 0);
        }
    }
}

impl Drop for LlamaRuntime {
    fn drop(&mut self) {
        unsafe {
            ffi::llama_free(self.ctx.as_ptr());
            ffi::llama_model_free(self.model.as_ptr());
        }
    }
}

fn thread_count() -> i32 {
    std::thread::available_parallelism()
        .map(|count| count.get() as i32)
        .unwrap_or(4)
}
