plugins {
    id("com.android.application")
}

android {
    namespace = "app.astracode.scanner"
    compileSdk = 36

    defaultConfig {
        applicationId = "app.astracode.scanner"
        minSdk = 26
        targetSdk = 36
        versionCode = 2
        versionName = "0.2"
        // The native scanner is built for 64-bit ARM, which is every current phone.
        // To run on an emulator, build it for x86_64 as well: ABIS="arm64-v8a x86_64" ./build-rust.sh
        ndk { abiFilters += "arm64-v8a" }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
}

dependencies {
    val camerax = "1.4.2"
    implementation("androidx.camera:camera-core:$camerax")
    implementation("androidx.camera:camera-camera2:$camerax")
    implementation("androidx.camera:camera-lifecycle:$camerax")
    implementation("androidx.camera:camera-view:$camerax")
    implementation("androidx.activity:activity-ktx:1.10.1")
    implementation("androidx.core:core-ktx:1.15.0")

    testImplementation("junit:junit:4.13.2")
}

// The scanner is the Rust library in ../rust. build-rust.sh compiles it with the NDK and
// puts the result in src/main/jniLibs, where the Android build picks it up.
val buildRust by tasks.registering(Exec::class) {
    description = "Builds the Rust scanner for Android."
    val rust = rootDir.resolve("../rust")
    inputs.files(fileTree(rust) {
        include("Cargo.toml", "Cargo.lock", "astracode/**", "astracode-android/**")
        exclude("**/target/**")
    })
    inputs.file(rootDir.resolve("build-rust.sh"))
    outputs.dir(projectDir.resolve("src/main/jniLibs"))
    commandLine("sh", rootDir.resolve("build-rust.sh").path)
}
tasks.named("preBuild") { dependsOn(buildRust) }
