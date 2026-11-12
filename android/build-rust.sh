#!/bin/sh
# Build the Rust scanner for Android and put it where the app picks it up
# (app/src/main/jniLibs/<abi>/libastracode_android.so).
#
#   ./build-rust.sh                 # arm64 phones
#   ABIS="arm64-v8a x86_64" ./build-rust.sh    # and the emulator
#
# Needs the Android NDK (ANDROID_NDK_HOME, or the newest one under
# $ANDROID_HOME/ndk) and the Rust target for each ABI:
#   rustup target add aarch64-linux-android x86_64-linux-android
# Gradle runs this before every build; see app/build.gradle.kts.
set -eu

# Gradle may run this without the login shell's PATH.
PATH="$HOME/.cargo/bin:$PATH"

here=$(cd "$(dirname "$0")" && pwd)
sdk=${ANDROID_HOME:-$(sed -n 's/^sdk\.dir=//p' "$here/local.properties" 2>/dev/null)}
ndk=${ANDROID_NDK_HOME:-$(ls -d "$sdk"/ndk/* 2>/dev/null | sort -V | tail -1)}
[ -d "$ndk" ] || { echo "build-rust.sh: no Android NDK found; set ANDROID_NDK_HOME" >&2; exit 1; }
bin=$(ls -d "$ndk"/toolchains/llvm/prebuilt/*/bin | head -1)
api=26        # the app's minSdk

for abi in ${ABIS:-arm64-v8a}; do
    case $abi in
        arm64-v8a) target=aarch64-linux-android ;;
        x86_64) target=x86_64-linux-android ;;
        *) echo "build-rust.sh: unknown ABI $abi" >&2; exit 1 ;;
    esac
    var=$(echo "$target" | tr 'a-z-' 'A-Z_')
    # Link with the NDK's clang for this target and API level.
    export "CARGO_TARGET_${var}_LINKER=$bin/${target}${api}-clang"
    export "CC_$(echo "$target" | tr - _)=$bin/${target}${api}-clang"
    export "AR_$(echo "$target" | tr - _)=$bin/llvm-ar"
    (cd "$here/../rust" && cargo build --release -p astracode-android --target "$target")
    mkdir -p "$here/app/src/main/jniLibs/$abi"
    cp "$here/../rust/target/$target/release/libastracode_android.so" "$here/app/src/main/jniLibs/$abi/"
done
