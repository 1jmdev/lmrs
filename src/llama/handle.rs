use std::ffi::{CStr, CString};
use std::path::Path;
use std::ptr::NonNull;

use hashbrown::HashSet;
use smallvec::SmallVec;
use tracing::{debug, instrument};

use crate::api::Message;
use crate::error::LmrsError;
use crate::llama::backend::ensure_backend_initialized;
use crate::llama::ffi;

pub type Token = ffi::llama_token;

#[derive(Clone, Copy, Debug)]
pub struct LoadConfig {
    pub context_size: u32,
    pub batch_size: u32,
    pub gpu_layers: i32,
    pub threads: i32,
    pub threads_batch: i32,
}

impl Default for LoadConfig {
    fn default() -> Self {
        let threads = default_threads();
        Self {
            context_size: 131_072,
            batch_size: 2048,
            gpu_layers: -1,
            threads,
            threads_batch: threads,
        }
    }
}

pub struct LlamaHandle {
    model: NonNull<ffi::llama_model>,
    ctx: NonNull<ffi::llama_context>,
    vocab: *const ffi::llama_vocab,
    vocab_size: usize,
    eos_token: Token,
    eot_token: Token,
}

impl LlamaHandle {
    #[instrument(skip(config), fields(model_path = %model_path.display()))]
    pub fn load(model_path: &Path, config: LoadConfig) -> Result<Self, LmrsError> {
        ensure_backend_initialized();

        let path = model_path
            .to_str()
            .ok_or_else(|| LmrsError::ModelPathNotUtf8(model_path.to_path_buf()))?;
        let c_path = CString::new(path).map_err(|_| LmrsError::CStringNul)?;

        let mut model_params = unsafe { ffi::llama_model_default_params() };
        model_params.n_gpu_layers = config.gpu_layers;

        let model = unsafe { ffi::llama_model_load_from_file(c_path.as_ptr(), model_params) };
        let model = match NonNull::new(model) {
            Some(model) => model,
            None => {
                if config.gpu_layers == 0 {
                    return Err(LmrsError::ModelLoadFailed);
                }

                model_params.n_gpu_layers = 0;
                let cpu_model =
                    unsafe { ffi::llama_model_load_from_file(c_path.as_ptr(), model_params) };
                let cpu_model = NonNull::new(cpu_model).ok_or(LmrsError::ModelLoadFailed)?;
                debug!(
                    requested_gpu_layers = config.gpu_layers,
                    "failed to load model with GPU offload; retried with CPU"
                );
                cpu_model
            }
        };

        let vocab = unsafe { ffi::llama_model_get_vocab(model.as_ptr()) };
        let vocab_size = unsafe { ffi::llama_vocab_n_tokens(vocab) as usize };
        let eos_token = unsafe { ffi::llama_vocab_eos(vocab) };
        let eot_token = unsafe { ffi::llama_vocab_eot(vocab) };

        let mut ctx_sizes: SmallVec<[u32; 8]> = SmallVec::new();
        ctx_sizes.push(config.context_size);
        if config.context_size > 8192 {
            let mut candidate = config.context_size / 2;
            while candidate >= 8192 {
                if !ctx_sizes.contains(&candidate) {
                    ctx_sizes.push(candidate);
                }
                candidate /= 2;
            }
            if !ctx_sizes.contains(&8192) {
                ctx_sizes.push(8192);
            }
        }

        let mut ctx = None;
        for context_size in ctx_sizes {
            let mut ctx_params = unsafe { ffi::llama_context_default_params() };
            ctx_params.n_ctx = context_size;
            ctx_params.n_batch = config.batch_size;
            ctx_params.n_seq_max = 1;
            ctx_params.n_threads = config.threads;
            ctx_params.n_threads_batch = config.threads_batch;

            let candidate_ctx = unsafe { ffi::llama_init_from_model(model.as_ptr(), ctx_params) };
            if let Some(candidate_ctx) = NonNull::new(candidate_ctx) {
                if context_size != config.context_size {
                    debug!(
                        requested_context_size = config.context_size,
                        selected_context_size = context_size,
                        "failed to initialize requested context size; using reduced context"
                    );
                }
                ctx = Some(candidate_ctx);
                break;
            }
        }

        let ctx = match ctx {
            Some(ctx) => ctx,
            None => {
                unsafe { ffi::llama_model_free(model.as_ptr()) };
                return Err(LmrsError::ContextInitFailed);
            }
        };

        let handle = Self {
            model,
            ctx,
            vocab,
            vocab_size,
            eos_token,
            eot_token,
        };
        debug!(vocab_size = handle.vocab_size, "llama context initialized");
        Ok(handle)
    }

    pub fn count_tokens(&self, text: &str) -> Result<usize, LmrsError> {
        self.tokenize(text).map(|tokens| tokens.len())
    }

    pub fn context_size(&self) -> usize {
        unsafe { ffi::llama_n_ctx(self.ctx.as_ptr()) as usize }
    }

    pub fn tokenize(&self, text: &str) -> Result<Vec<Token>, LmrsError> {
        let mut tokens = vec![0; text.len() + 8];
        let mut count = unsafe {
            ffi::llama_tokenize(
                self.vocab,
                text.as_ptr().cast(),
                text.len() as i32,
                tokens.as_mut_ptr(),
                tokens.len() as i32,
                true,
                true,
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
                    true,
                )
            };
        }

        if count < 0 {
            return Err(LmrsError::TokenizeFailed(count));
        }

        tokens.truncate(count as usize);
        Ok(tokens)
    }

    pub fn decode(&self, tokens: &mut [Token]) -> Result<(), LmrsError> {
        let batch = unsafe { ffi::llama_batch_get_one(tokens.as_mut_ptr(), tokens.len() as i32) };
        let result = unsafe { ffi::llama_decode(self.ctx.as_ptr(), batch) };
        if result == 0 {
            Ok(())
        } else {
            Err(LmrsError::DecodeFailed(result))
        }
    }

    pub fn logits(&self) -> Result<&[f32], LmrsError> {
        let logits = unsafe { ffi::llama_get_logits_ith(self.ctx.as_ptr(), -1) };
        let logits = NonNull::new(logits).ok_or(LmrsError::NoLogits)?;
        let logits = unsafe { std::slice::from_raw_parts(logits.as_ptr(), self.vocab_size) };
        Ok(logits)
    }

    pub fn token_is_eog(&self, token: Token) -> bool {
        if unsafe { ffi::llama_vocab_is_eog(self.vocab, token) } {
            return true;
        }

        (self.eos_token != ffi::LLAMA_TOKEN_NULL && token == self.eos_token)
            || (self.eot_token != ffi::LLAMA_TOKEN_NULL && token == self.eot_token)
    }

    pub fn token_to_piece_bytes(&self, token: Token) -> Result<Vec<u8>, LmrsError> {
        let mut scratch: SmallVec<[u8; 64]> = SmallVec::with_capacity(64);
        scratch.resize(64, 0);
        loop {
            let written = unsafe {
                ffi::llama_token_to_piece(
                    self.vocab,
                    token,
                    scratch.as_mut_ptr().cast(),
                    scratch.len() as i32,
                    0,
                    true,
                )
            };

            if written >= 0 && written as usize <= scratch.len() {
                scratch.truncate(written as usize);
                return Ok(scratch.into_vec());
            }

            if written < 0 {
                scratch.resize((-written) as usize, 0);
                let retry = unsafe {
                    ffi::llama_token_to_piece(
                        self.vocab,
                        token,
                        scratch.as_mut_ptr().cast(),
                        scratch.len() as i32,
                        0,
                        true,
                    )
                };
                if retry < 0 {
                    return Err(LmrsError::TokenToPieceFailed(retry));
                }
                scratch.truncate(retry as usize);
                return Ok(scratch.into_vec());
            }

            scratch.resize(written as usize, 0);
        }
    }

    pub fn apply_chat_template(
        &self,
        messages: &[Message],
        enable_thinking: bool,
    ) -> Result<String, LmrsError> {
        let roles = messages
            .iter()
            .map(|message| CString::new(message.role.as_str()).map_err(|_| LmrsError::CStringNul))
            .collect::<Result<Vec<_>, _>>()?;
        let contents = messages
            .iter()
            .map(|message| {
                CString::new(message.content.replace('\0', "")).map_err(|_| LmrsError::CStringNul)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let chat_messages = roles
            .iter()
            .zip(contents.iter())
            .map(|(role, content)| ffi::llama_chat_message {
                role: role.as_ptr(),
                content: content.as_ptr(),
            })
            .collect::<Vec<_>>();

        let estimate = messages
            .iter()
            .map(|message| message.role.len() + message.content.len() + 64)
            .sum::<usize>()
            .max(512);

        let mut template_names = Vec::with_capacity(8);
        let mut seen_templates = HashSet::with_capacity(8);
        unsafe {
            let ptr = ffi::llama_model_chat_template(self.model.as_ptr(), std::ptr::null());
            if !ptr.is_null() {
                let name = CString::new(CStr::from_ptr(ptr).to_bytes())
                    .map_err(|_| LmrsError::CStringNul)?;
                seen_templates.insert(name.as_bytes().to_vec());
                template_names.push(name);
            }
        }
        unsafe {
            let count = ffi::llama_chat_builtin_templates(std::ptr::null_mut(), 0);
            if count > 0 {
                let mut builtin_templates = vec![std::ptr::null(); count as usize];
                let copied = ffi::llama_chat_builtin_templates(
                    builtin_templates.as_mut_ptr(),
                    builtin_templates.len(),
                );
                let copied = copied.clamp(0, builtin_templates.len() as i32) as usize;
                for template_ptr in builtin_templates.into_iter().take(copied) {
                    if template_ptr.is_null() {
                        continue;
                    }
                    let template = CString::new(CStr::from_ptr(template_ptr).to_bytes())
                        .map_err(|_| LmrsError::CStringNul)?;
                    if seen_templates.insert(template.as_bytes().to_vec()) {
                        template_names.push(template);
                    }
                }
            }
        }

        for template in template_names {
            let template_supports_thinking = template
                .as_bytes()
                .windows(THINK_TAG.len())
                .any(|window| window == THINK_TAG);
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

            if written >= 0 && written as usize <= buffer.len() {
                let mut rendered =
                    String::from_utf8_lossy(&buffer[..written as usize]).into_owned();
                if enable_thinking && template_supports_thinking {
                    ensure_thinking_prompt(&mut rendered);
                }
                if !rendered.is_empty() {
                    return Ok(rendered);
                }
            }

            if written > 0 {
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
                if rewritten >= 0 && rewritten as usize <= buffer.len() {
                    let mut rendered =
                        String::from_utf8_lossy(&buffer[..rewritten as usize]).into_owned();
                    if enable_thinking && template_supports_thinking {
                        ensure_thinking_prompt(&mut rendered);
                    }
                    if !rendered.is_empty() {
                        return Ok(rendered);
                    }
                }
            }
        }

        Err(LmrsError::TemplateApplyFailed)
    }
}

const THINK_TAG: &[u8] = b"<think>";

fn ensure_thinking_prompt(rendered: &mut String) {
    if rendered.trim_end().ends_with("<think>") {
        return;
    }

    rendered.push('\n');
    rendered.push_str("<think>\n");
}

impl Drop for LlamaHandle {
    fn drop(&mut self) {
        unsafe {
            ffi::llama_free(self.ctx.as_ptr());
            ffi::llama_model_free(self.model.as_ptr());
        }
    }
}

fn default_threads() -> i32 {
    std::thread::available_parallelism()
        .map(|threads| threads.get() as i32)
        .unwrap_or(4)
}
