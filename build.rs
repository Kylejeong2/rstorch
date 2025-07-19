
fn main() {
    cc::Build::new()
        .cpp(true)
        .file("src/csrc/tensor_impl.cpp")
        .include("src/csrc")
        .flag_if_supported("-std=c++14")
        .compile("tensor_backend");
    
    // Link the C++ standard library
    if cfg!(target_os = "linux") {
        println!("cargo:rustc-link-lib=stdc++");
    } else if cfg!(target_os = "macos") {
        println!("cargo:rustc-link-lib=c++");
    }
    
    // Rerun if C++ files change
    println!("cargo:rerun-if-changed=src/csrc/tensor_impl.cpp");
    println!("cargo:rerun-if-changed=src/csrc/tensor.h");
}