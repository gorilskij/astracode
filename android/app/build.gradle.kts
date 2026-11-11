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
        versionCode = 1
        versionName = "0.1"
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

    testOptions {
        unitTests.all {
            // Optional large fixture set: ASTRA_FIXTURES=/path/to/dir ./gradlew test
            it.environment("ASTRA_FIXTURES", System.getenv("ASTRA_FIXTURES") ?: "")
            it.testLogging.showStandardStreams = true
            it.maxHeapSize = "2g"
        }
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
