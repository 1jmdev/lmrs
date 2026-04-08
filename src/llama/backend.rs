use std::sync::Once;

use crate::llama::ffi;

static BACKEND_INIT: Once = Once::new();

unsafe extern "C" fn noop_log(
    _level: ffi::ggml_log_level,
    _text: *const std::os::raw::c_char,
    _user_data: *mut std::os::raw::c_void,
) {
}

pub fn ensure_backend_initialized() {
    BACKEND_INIT.call_once(|| unsafe {
        ffi::llama_log_set(Some(noop_log), std::ptr::null_mut());
        ffi::ggml_backend_load_all_from_path(std::ptr::null());
        ffi::llama_backend_init();
    });
}
