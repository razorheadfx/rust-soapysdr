extern crate bindgen;
extern crate pkg_config;

use std::env;
use std::path::PathBuf;

fn main() {
    if let Err(e) = pkg_config::Config::new()
        .atleast_version("0.7.0")
        .probe("SoapySDR")
    {
        panic!("Couldn't find SoapySDR: {}", e);
    }

    let bindings = bindgen::Builder::default()
        .trust_clang_mangling(false)
        .header("wrapper.h")
        .generate()
        .expect("Unable to generate bindings");

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("soapy_bindings.rs"))
        .expect("Couldn't write bindings!");
}
