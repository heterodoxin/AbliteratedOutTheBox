# Mac

Docker on a Mac runs Linux in a virtual machine that has no access to the Mac's GPU (no Metal), so a container could only
use the CPU, and Strata has no CPU-only mode worth running. Use the Mac app or the Terminal instead:

    curl -fsSL <release-folder>/install.sh | sh        # then: strata launch

See docs/MAC.md and docs/TERMINAL_LAUNCH.md. The same goes for Windows with Docker Desktop and an AMD or Intel card:
the WSL2 virtual machine passes NVIDIA cards through, not Radeon or Arc cards. Use the Windows app or Terminal.
