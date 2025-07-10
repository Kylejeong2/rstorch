use std::process::Command;
use clap::Parser;

/// Simple distributed runner that dispatches a Python script through `mpiexec`.
/// Mirrors the behaviour of the reference Python snippet.
#[derive(Parser, Debug)]
#[command(name = "rstorch-distributed-run", about = "Spawn multi-process training via mpiexec")] 
pub struct Args {
    /// Number of processes per node
    #[arg(long)]
    pub nproc_per_node: usize,

    /// Number of nodes (defaults to 1)
    #[arg(long, default_value_t = 1)]
    pub nnodes: usize,

    /// Path to the training script (executed with the system python)
    pub script: String,

    /// All remaining arguments are forwarded to the script
    #[arg(trailing_var_arg = true)]
    pub script_args: Vec<String>,
}

/// Entry-point you can call from `main()` or tests.
pub fn run() -> anyhow::Result<()> {
    let args = Args::parse();

    let world_size = args.nproc_per_node * args.nnodes;

    // Build: mpiexec -n <world_size> --allow-run-as-root python <script> <script_args..>
    let mut cmd = Command::new("mpiexec");
    cmd.arg("-n")
        .arg(world_size.to_string())
        .arg("--allow-run-as-root")
        .arg("python")
        .arg(&args.script)
        .args(&args.script_args);

    let status = cmd.status()?;
    if !status.success() {
        anyhow::bail!("mpiexec returned non-zero exit status: {}", status);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    // Just check that argument parsing works; we don't actually spawn mpiexec in CI.
    #[test]
    fn parse_cli() {
        let argv = [
            "prog",
            "--nproc_per_node", "4",
            "--nnodes", "2",
            "train.py",
            "--lr", "0.01",
        ];
        let args = Args::parse_from(&argv);
        assert_eq!(args.nproc_per_node, 4);
        assert_eq!(args.nnodes, 2);
        assert_eq!(args.script, "train.py");
        assert_eq!(args.script_args, vec!["--lr", "0.01"]);
    }
} 