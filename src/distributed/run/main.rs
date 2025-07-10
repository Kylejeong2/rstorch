/**
import sys
from . import run

def main():
    run.main()

if __name__ == "__main__":
    main()
*/

use rstorch::distributed::run;

fn main() {
    run::run().unwrap();
}