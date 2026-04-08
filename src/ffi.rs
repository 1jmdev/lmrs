#![allow(
    dead_code,
    non_camel_case_types,
    non_snake_case,
    non_upper_case_globals
)]
#![allow(unused)]
#![allow(unnecessary_transmutes)]
#![allow(clippy::all)]

include!(concat!(env!("OUT_DIR"), "/llama_bindings.rs"));

unsafe extern "C" {
    pub fn ggml_backend_load_all_from_path(dir_path: *const std::os::raw::c_char);
}
