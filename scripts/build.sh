#!/usr/bin/env zsh

# Ensure the script is sourced so the environment persists in your terminal
if [[ "${ZSH_EVAL_CONTEXT:-}" != *:file* ]]; then
    echo "Error: Run this script with 'source ${0}' (or '. ${0}') so ROS 2 stays sourced in your terminal." >&2
    exit 1
fi

build_openarm_ws() {
    local script_dir="${1:A:h}"
    local repo_root
    repo_root="$(git -C "$script_dir" rev-parse --show-toplevel)" || return 1

    local orig_dir="$PWD"

    source /opt/ros/jazzy/setup.zsh || return 1

    cd "$repo_root/ros2_ws" || return 1

    colcon build \
        --symlink-install \
        --packages-select \
        openarm_description \
        openarm_hardware \
        openarm_bringup \
        me6404_bringup \
        || {
        cd "$orig_dir"
        return 1
    }

    source "$repo_root/ros2_ws/install/setup.zsh"
    cd "$orig_dir"
    echo "Build complete and ROS 2 workspace sourced."
}

build_openarm_ws "$0"
