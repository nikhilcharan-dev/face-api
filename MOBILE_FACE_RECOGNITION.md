# MOBILE_FACE_RECOGNITION.md

## Mobile Face Capture Screen — Attendance System

> Complete implementation guide for building the real-time face capture and recognition screen
> that communicates with the Jetson Nano / GPU VM backend (FastAPI + AntelopeV2 + FAISS).

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Tech Stack & Dependencies](#2-tech-stack--dependencies)
3. [Project Structure](#3-project-structure)
4. [Camera Module — Live Preview](#4-camera-module--live-preview)
5. [Face Detection — On-Device ML](#5-face-detection--on-device-ml)
6. [Face Crop & Preprocessing](#6-face-crop--preprocessing)
7. [Network Layer — API Client](#7-network-layer--api-client)
8. [UI/UX — Face Capture Screen](#8-uiux--face-capture-screen)
9. [State Management](#9-state-management)
10. [Attendance Flow & Business Logic](#10-attendance-flow--business-logic)
11. [Error Handling & Edge Cases](#11-error-handling--edge-cases)
12. [Performance Optimization](#12-performance-optimization)
13. [Security Considerations](#13-security-considerations)
14. [Testing Strategy](#14-testing-strategy)
15. [Full Code Implementation](#15-full-code-implementation)

---

## 1. Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    MOBILE DEVICE                         │
│                                                         │
│  ┌──────────┐    ┌──────────────┐    ┌───────────────┐  │
│  │  Camera   │───▶│ Face Detect  │───▶│  Crop + JPEG  │  │
│  │  Preview  │    │ (on-device)  │    │  Compress     │  │
│  │  25 FPS   │    │  ML Kit /    │    │  ~80KB        │  │
│  │           │    │  MediaPipe   │    │  payload      │  │
│  └──────────┘    └──────────────┘    └──────┬────────┘  │
│                                             │           │
│  ┌──────────────────────────────────────────┘           │
│  │                                                      │
│  ▼                                                      │
│  ┌──────────────┐    ┌──────────────┐                   │
│  │  API Client   │───▶│  Result UI   │                   │
│  │  POST /detect │    │  ✓ Name      │                   │
│  │  timeout: 10s │    │  ✓ Conf %    │                   │
│  │               │◀───│  ✓ Time      │                   │
│  └──────┬───────┘    └──────────────┘                   │
│         │                                               │
└─────────┼───────────────────────────────────────────────┘
          │  HTTPS / VPN
          ▼
┌─────────────────────────────────────────────────────────┐
│              JETSON NANO / GPU VM                        │
│                                                         │
│  Nginx ──▶ FastAPI ──▶ AntelopeV2 ──▶ FAISS ──▶ Result  │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### Data Flow Summary

| Step | Location | Action | Latency |
|------|----------|--------|---------|
| 1 | Mobile | Camera frame captured at 25 FPS | ~1ms |
| 2 | Mobile | Face detection via on-device ML | 5–10ms |
| 3 | Mobile | Bounding box crop + JPEG compress | ~2ms |
| 4 | Network | POST cropped face to `/api/v1/detect` | ~5ms |
| 5 | Server | AntelopeV2 generates 512-dim embedding | 10–15ms |
| 6 | Server | FAISS nearest neighbor search | 1–2ms |
| 7 | Mobile | Display identity result + confidence | ~1ms |

**Total round-trip: ~25ms** (on local network)

---

## 2. Tech Stack & Dependencies

### React Native (Recommended)

```json
{
  "dependencies": {
    "react-native": ">=0.73.0",
    "react-native-vision-camera": "^4.0.0",
    "react-native-worklets-core": "^1.0.0",
    "react-native-face-detection": "^2.0.0",
    "@react-native-ml-kit/face-detection": "^1.3.0",
    "react-native-reanimated": "^3.6.0",
    "axios": "^1.6.0",
    "zustand": "^4.5.0",
    "react-native-haptic-feedback": "^2.2.0",
    "react-native-svg": "^14.0.0",
    "@react-navigation/native": "^6.0.0",
    "react-native-mmkv": "^2.11.0"
  },
  "devDependencies": {
    "typescript": "^5.3.0",
    "@types/react-native": "^0.73.0",
    "jest": "^29.0.0",
    "@testing-library/react-native": "^12.0.0"
  }
}
```

### Flutter (Alternative)

```yaml
dependencies:
  camera: ^0.11.0
  google_mlkit_face_detection: ^0.11.0
  http: ^1.2.0
  image: ^4.1.0
  provider: ^6.1.0
  flutter_animate: ^4.4.0
```

### Native Android (Kotlin)

```kotlin
// build.gradle.kts
dependencies {
    implementation("androidx.camera:camera-camera2:1.3.1")
    implementation("androidx.camera:camera-lifecycle:1.3.1")
    implementation("androidx.camera:camera-view:1.3.1")
    implementation("com.google.mlkit:face-detection:16.1.6")
    implementation("com.squareup.retrofit2:retrofit:2.9.0")
    implementation("com.squareup.okhttp3:okhttp:4.12.0")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.7.3")
}
```

> **This guide uses React Native with VisionCamera as the primary implementation.**
> All concepts translate directly to Flutter and native platforms.

---

## 3. Project Structure

```
src/
├── screens/
│   └── FaceCaptureScreen.tsx          # Main capture screen
├── components/
│   ├── CameraPreview.tsx              # Camera with overlay
│   ├── FaceOverlay.tsx                # Bounding box + guide oval
│   ├── CaptureButton.tsx              # Animated capture trigger
│   ├── ResultCard.tsx                 # Identity result display
│   ├── AttendanceStatusBar.tsx        # Top status indicator
│   └── FaceGuideFrame.tsx             # Oval alignment guide
├── hooks/
│   ├── useFaceDetection.ts            # Face detection logic
│   ├── useCamera.ts                   # Camera lifecycle
│   ├── useFaceCrop.ts                 # Crop + compress logic
│   └── useAttendance.ts              # Full attendance flow
├── services/
│   ├── api.ts                         # Axios client + interceptors
│   ├── faceApi.ts                     # /detect endpoint wrapper
│   └── offlineQueue.ts               # Offline request queue
├── utils/
│   ├── imageProcessing.ts            # Crop, resize, compress
│   ├── faceValidation.ts             # Quality checks
│   └── constants.ts                  # Config values
├── store/
│   └── attendanceStore.ts            # Zustand state
├── types/
│   └── index.ts                      # TypeScript interfaces
└── __tests__/
    ├── FaceCaptureScreen.test.tsx
    ├── useFaceDetection.test.ts
    └── api.test.ts
```

---

## 4. Camera Module — Live Preview

### Camera Setup with VisionCamera v4

```typescript
// hooks/useCamera.ts
import { useRef, useCallback, useState } from 'react';
import {
  Camera,
  useCameraDevice,
  useCameraPermission,
  useCameraFormat,
  PhotoFile,
  CameraPosition,
} from 'react-native-vision-camera';

interface UseCameraResult {
  cameraRef: React.RefObject<Camera>;
  device: ReturnType<typeof useCameraDevice>;
  hasPermission: boolean;
  requestPermission: () => Promise<boolean>;
  format: ReturnType<typeof useCameraFormat>;
  isActive: boolean;
  setIsActive: (active: boolean) => void;
  position: CameraPosition;
  flipCamera: () => void;
  takeSnapshot: () => Promise<PhotoFile | null>;
}

export function useCamera(): UseCameraResult {
  const cameraRef = useRef<Camera>(null);
  const [isActive, setIsActive] = useState(true);
  const [position, setPosition] = useState<CameraPosition>('front');
  const { hasPermission, requestPermission } = useCameraPermission();

  const device = useCameraDevice(position, {
    physicalDevices: ['wide-angle-camera'],
  });

  // Target 640x480 for fast processing — NOT full resolution
  const format = useCameraFormat(device, [
    { videoResolution: { width: 640, height: 480 } },
    { fps: 30 },
  ]);

  const flipCamera = useCallback(() => {
    setPosition((prev) => (prev === 'front' ? 'back' : 'front'));
  }, []);

  const takeSnapshot = useCallback(async (): Promise<PhotoFile | null> => {
    if (!cameraRef.current) return null;
    try {
      const photo = await cameraRef.current.takePhoto({
        qualityPrioritization: 'speed',
        flash: 'off',
        enableShutterSound: false,
      });
      return photo;
    } catch (error) {
      console.error('[Camera] Snapshot failed:', error);
      return null;
    }
  }, []);

  return {
    cameraRef,
    device,
    hasPermission,
    requestPermission,
    format,
    isActive,
    setIsActive,
    position,
    flipCamera,
    takeSnapshot,
  };
}
```

### Camera Preview Component

```typescript
// components/CameraPreview.tsx
import React, { useCallback } from 'react';
import { StyleSheet, View } from 'react-native';
import {
  Camera,
  useFrameProcessor,
} from 'react-native-vision-camera';
import { useCamera } from '../hooks/useCamera';
import { FaceOverlay } from './FaceOverlay';
import { FaceGuideFrame } from './FaceGuideFrame';

interface CameraPreviewProps {
  onFaceDetected: (face: DetectedFace) => void;
  onFrameCapture: (frame: FrameData) => void;
  isProcessing: boolean;
}

export function CameraPreview({
  onFaceDetected,
  onFrameCapture,
  isProcessing,
}: CameraPreviewProps) {
  const {
    cameraRef,
    device,
    hasPermission,
    format,
    isActive,
  } = useCamera();

  // Frame processor runs on every camera frame (JS worklet thread)
  const frameProcessor = useFrameProcessor((frame) => {
    'worklet';
    // Face detection runs here — see Section 5
    // This runs at ~25 FPS on the GPU thread, NOT the JS thread
  }, []);

  if (!device || !hasPermission) {
    return <CameraPermissionPlaceholder />;
  }

  return (
    <View style={styles.container}>
      <Camera
        ref={cameraRef}
        style={StyleSheet.absoluteFill}
        device={device}
        format={format}
        isActive={isActive && !isProcessing}
        frameProcessor={frameProcessor}
        pixelFormat="yuv"
        photo={true}
        orientation="portrait"
        enableFpsGraph={__DEV__}
      />
      <FaceGuideFrame />
      <FaceOverlay />
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#000',
  },
});
```

---

## 5. Face Detection — On-Device ML

### Using ML Kit Face Detection (via Frame Processor)

```typescript
// hooks/useFaceDetection.ts
import { useState, useCallback, useRef } from 'react';
import { runOnJS } from 'react-native-reanimated';

// Face detection model options
const FACE_DETECTION_OPTIONS = {
  performanceMode: 'fast' as const,       // 'fast' or 'accurate'
  landmarkMode: 'none' as const,          // We don't need landmarks
  contourMode: 'none' as const,           // We don't need contours
  classificationMode: 'all' as const,     // Eyes open, smiling
  minFaceSize: 0.25,                      // Min 25% of frame
  enableTracking: true,                   // Track face ID across frames
};

export interface DetectedFace {
  bounds: {
    x: number;      // Top-left X (0-1 normalized)
    y: number;      // Top-left Y (0-1 normalized)
    width: number;   // Width (0-1 normalized)
    height: number;  // Height (0-1 normalized)
  };
  trackingId: number;
  rightEyeOpenProbability: number;
  leftEyeOpenProbability: number;
  smilingProbability: number;
  rollAngle: number;    // Head tilt (degrees)
  yawAngle: number;     // Head left/right turn (degrees)
  pitchAngle: number;   // Head up/down tilt (degrees)
}

export interface FaceValidation {
  isValid: boolean;
  issues: string[];
  quality: 'good' | 'fair' | 'poor';
}

export function useFaceDetection() {
  const [detectedFaces, setDetectedFaces] = useState<DetectedFace[]>([]);
  const [validation, setValidation] = useState<FaceValidation>({
    isValid: false,
    issues: [],
    quality: 'poor',
  });
  const lastDetectionTime = useRef(0);
  const DETECTION_THROTTLE_MS = 40; // ~25 FPS max

  /**
   * Validate face quality before sending to server.
   * This prevents wasted API calls with poor images.
   */
  const validateFace = useCallback((face: DetectedFace): FaceValidation => {
    const issues: string[] = [];

    // --- SIZE CHECK ---
    // Face must be at least 25% of frame width
    if (face.bounds.width < 0.25) {
      issues.push('Move closer — face too small');
    }
    // Face shouldn't be more than 80% of frame (too close)
    if (face.bounds.width > 0.80) {
      issues.push('Move back — face too close');
    }

    // --- POSITION CHECK ---
    // Face center should be within middle 60% of frame
    const centerX = face.bounds.x + face.bounds.width / 2;
    const centerY = face.bounds.y + face.bounds.height / 2;
    if (centerX < 0.2 || centerX > 0.8) {
      issues.push('Center your face horizontally');
    }
    if (centerY < 0.2 || centerY > 0.8) {
      issues.push('Center your face vertically');
    }

    // --- ANGLE CHECK ---
    // Face should be roughly frontal (within ±15° on each axis)
    if (Math.abs(face.yawAngle) > 15) {
      issues.push('Face the camera directly');
    }
    if (Math.abs(face.pitchAngle) > 15) {
      issues.push('Level your head — tilt detected');
    }
    if (Math.abs(face.rollAngle) > 15) {
      issues.push('Straighten your head');
    }

    // --- EYE CHECK ---
    // Both eyes should be open (anti-spoof basic check)
    if (face.rightEyeOpenProbability < 0.5 || face.leftEyeOpenProbability < 0.5) {
      issues.push('Open both eyes');
    }

    // --- QUALITY SCORING ---
    let quality: 'good' | 'fair' | 'poor';
    if (issues.length === 0) {
      quality = 'good';
    } else if (issues.length <= 2) {
      quality = 'fair';
    } else {
      quality = 'poor';
    }

    return {
      isValid: issues.length === 0,
      issues,
      quality,
    };
  }, []);

  /**
   * Called from frame processor on every camera frame.
   * Throttled to ~25 FPS to avoid overloading the JS thread.
   */
  const onFaceDetectionResult = useCallback(
    (faces: DetectedFace[]) => {
      const now = Date.now();
      if (now - lastDetectionTime.current < DETECTION_THROTTLE_MS) return;
      lastDetectionTime.current = now;

      setDetectedFaces(faces);

      if (faces.length === 1) {
        const result = validateFace(faces[0]);
        setValidation(result);
      } else if (faces.length === 0) {
        setValidation({ isValid: false, issues: ['No face detected'], quality: 'poor' });
      } else {
        setValidation({ isValid: false, issues: ['Multiple faces detected — only one allowed'], quality: 'poor' });
      }
    },
    [validateFace]
  );

  return {
    detectedFaces,
    validation,
    onFaceDetectionResult,
  };
}
```

### Frame Processor Plugin (Worklet)

```typescript
// plugins/faceDetectionPlugin.ts
// This runs on the native camera thread — NOT the JS thread

import { VisionCameraProxy, Frame } from 'react-native-vision-camera';

const plugin = VisionCameraProxy.initFrameProcessorPlugin(
  'detectFaces',
  {}
);

export function detectFaces(frame: Frame): DetectedFace[] {
  'worklet';
  if (!plugin) return [];

  const result = plugin.call(frame, {
    performanceMode: 'fast',
    minFaceSize: 0.25,
    enableTracking: true,
  });

  return (result as DetectedFace[]) ?? [];
}
```

### Native Module (Android — Kotlin)

```kotlin
// android/.../FaceDetectionFrameProcessorPlugin.kt
package com.yourapp.frameprocessors

import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.face.FaceDetection
import com.google.mlkit.vision.face.FaceDetectorOptions
import com.mrousavy.camera.frameprocessors.Frame
import com.mrousavy.camera.frameprocessors.FrameProcessorPlugin
import com.mrousavy.camera.frameprocessors.VisionCameraProxy

class FaceDetectionPlugin(proxy: VisionCameraProxy, options: Map<String, Any>?) :
    FrameProcessorPlugin(proxy, options) {

    private val detector = FaceDetection.getClient(
        FaceDetectorOptions.Builder()
            .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_FAST)
            .setLandmarkMode(FaceDetectorOptions.LANDMARK_MODE_NONE)
            .setClassificationMode(FaceDetectorOptions.CLASSIFICATION_MODE_ALL)
            .setMinFaceSize(0.25f)
            .enableTracking()
            .build()
    )

    override fun callback(frame: Frame, arguments: Map<String, Any>?): Any {
        val mediaImage = frame.image ?: return emptyList<Map<String, Any>>()
        val image = InputImage.fromMediaImage(
            mediaImage,
            frame.imageProxy.imageInfo.rotationDegrees
        )

        // Synchronous detection (blocks frame processor thread, not UI thread)
        val task = detector.process(image)
        val faces = com.google.android.gms.tasks.Tasks.await(task)

        val frameWidth = frame.width.toFloat()
        val frameHeight = frame.height.toFloat()

        return faces.map { face ->
            val bounds = face.boundingBox
            mapOf(
                "bounds" to mapOf(
                    "x" to bounds.left / frameWidth,
                    "y" to bounds.top / frameHeight,
                    "width" to bounds.width() / frameWidth,
                    "height" to bounds.height() / frameHeight
                ),
                "trackingId" to (face.trackingId ?: -1),
                "rightEyeOpenProbability" to (face.rightEyeOpenProbability ?: 0f),
                "leftEyeOpenProbability" to (face.leftEyeOpenProbability ?: 0f),
                "smilingProbability" to (face.smilingProbability ?: 0f),
                "rollAngle" to face.headEulerAngleZ,
                "yawAngle" to face.headEulerAngleY,
                "pitchAngle" to face.headEulerAngleX
            )
        }
    }
}
```

### Native Module (iOS — Swift)

```swift
// ios/.../FaceDetectionPlugin.swift
import MLKitFaceDetection
import MLKitVision
import VisionCamera

@objc(FaceDetectionPlugin)
public class FaceDetectionPlugin: FrameProcessorPlugin {
    private lazy var detector: FaceDetector = {
        let options = FaceDetectorOptions()
        options.performanceMode = .fast
        options.classificationMode = .all
        options.isTrackingEnabled = true
        options.minFaceSize = 0.25
        return FaceDetector.faceDetector(options: options)
    }()

    public override func callback(
        _ frame: Frame,
        withArguments arguments: [String: Any]?
    ) -> Any? {
        guard let buffer = CMSampleBufferGetImageBuffer(frame.buffer) else {
            return []
        }

        let image = VisionImage(buffer: frame.buffer)
        image.orientation = frame.orientation

        do {
            let faces = try detector.results(in: image)
            let frameWidth = CGFloat(CVPixelBufferGetWidth(buffer))
            let frameHeight = CGFloat(CVPixelBufferGetHeight(buffer))

            return faces.map { face in
                let bounds = face.frame
                return [
                    "bounds": [
                        "x": bounds.origin.x / frameWidth,
                        "y": bounds.origin.y / frameHeight,
                        "width": bounds.size.width / frameWidth,
                        "height": bounds.size.height / frameHeight,
                    ],
                    "trackingId": face.trackingID,
                    "rightEyeOpenProbability": face.rightEyeOpenProbability,
                    "leftEyeOpenProbability": face.leftEyeOpenProbability,
                    "smilingProbability": face.smilingProbability,
                    "rollAngle": face.headEulerAngleZ,
                    "yawAngle": face.headEulerAngleY,
                    "pitchAngle": face.headEulerAngleX,
                ] as [String: Any]
            }
        } catch {
            return []
        }
    }
}
```

---

## 6. Face Crop & Preprocessing

```typescript
// utils/imageProcessing.ts
import { Image } from 'react-native';
import RNFS from 'react-native-fs';

interface CropConfig {
  padding: number;       // Padding around face (fraction of face size)
  targetSize: number;    // Output square size in pixels
  jpegQuality: number;   // 0-100 JPEG compression quality
}

const DEFAULT_CROP_CONFIG: CropConfig = {
  padding: 0.3,          // 30% padding around detected face
  targetSize: 224,        // 224x224 — standard face model input
  jpegQuality: 85,        // Good quality, ~80KB output
};

/**
 * Crop the face region from a full camera frame.
 *
 * Why crop on mobile?
 * - Reduces payload from ~2MB (full frame) to ~80KB (cropped face)
 * - Server skips face detection step entirely
 * - 96% bandwidth savings
 * - Faster server-side processing
 */
export async function cropFaceFromPhoto(
  photoPath: string,
  faceBounds: {
    x: number;
    y: number;
    width: number;
    height: number;
  },
  config: CropConfig = DEFAULT_CROP_CONFIG
): Promise<{ base64: string; filePath: string; sizeKB: number }> {

  // Calculate crop region with padding
  const padX = faceBounds.width * config.padding;
  const padY = faceBounds.height * config.padding;

  const cropX = Math.max(0, faceBounds.x - padX);
  const cropY = Math.max(0, faceBounds.y - padY);
  const cropW = Math.min(1 - cropX, faceBounds.width + padX * 2);
  const cropH = Math.min(1 - cropY, faceBounds.height + padY * 2);

  // Make it square (take the larger dimension)
  const squareSize = Math.max(cropW, cropH);
  const squareCropX = cropX - (squareSize - cropW) / 2;
  const squareCropY = cropY - (squareSize - cropH) / 2;

  // Use native image manipulation (ImageEditor or react-native-image-crop-tools)
  const croppedPath = await cropImage(photoPath, {
    x: squareCropX,
    y: squareCropY,
    width: squareSize,
    height: squareSize,
    targetWidth: config.targetSize,
    targetHeight: config.targetSize,
    quality: config.jpegQuality,
  });

  // Read as base64 for API payload
  const base64 = await RNFS.readFile(croppedPath, 'base64');
  const stats = await RNFS.stat(croppedPath);
  const sizeKB = Math.round(stats.size / 1024);

  return {
    base64,
    filePath: croppedPath,
    sizeKB,
  };
}

/**
 * Quick brightness / contrast check on the cropped face.
 * Rejects images that are too dark or blown out.
 */
export function checkImageQuality(base64: string): {
  isAcceptable: boolean;
  brightness: 'too_dark' | 'ok' | 'too_bright';
  isBlurry: boolean;
} {
  // Decode a small sample of pixels to estimate brightness
  // In production, use a native module for speed
  const pixelSample = decodeBase64Sample(base64, 100); // sample 100 pixels
  const avgBrightness = pixelSample.reduce((sum, px) => sum + px.luminance, 0) / pixelSample.length;

  let brightness: 'too_dark' | 'ok' | 'too_bright';
  if (avgBrightness < 40) {
    brightness = 'too_dark';
  } else if (avgBrightness > 220) {
    brightness = 'too_bright';
  } else {
    brightness = 'ok';
  }

  // Blur detection via variance of Laplacian (simplified)
  const variance = calculateVariance(pixelSample.map(px => px.luminance));
  const isBlurry = variance < 100; // Low variance = blurry

  return {
    isAcceptable: brightness === 'ok' && !isBlurry,
    brightness,
    isBlurry,
  };
}
```

---

## 7. Network Layer — API Client

### Base API Client

```typescript
// services/api.ts
import axios, { AxiosInstance, AxiosError, InternalAxiosRequestConfig } from 'axios';
import { MMKV } from 'react-native-mmkv';

const storage = new MMKV();

// Server configuration
const API_CONFIG = {
  // Change this based on deployment:
  // Jetson Nano (local):   'http://192.168.1.100:8080'
  // GPU VM (cloud):        'https://api.attendance.yourcompany.com'
  BASE_URL: __DEV__
    ? 'http://192.168.1.100:8080'
    : 'https://api.attendance.yourcompany.com',

  TIMEOUT: 10_000,          // 10 second timeout
  RETRY_COUNT: 2,           // Retry twice on failure
  RETRY_DELAY: 500,         // 500ms between retries
};

function createApiClient(): AxiosInstance {
  const client = axios.create({
    baseURL: API_CONFIG.BASE_URL,
    timeout: API_CONFIG.TIMEOUT,
    headers: {
      'Content-Type': 'application/json',
      'Accept': 'application/json',
    },
  });

  // --- REQUEST INTERCEPTOR ---
  client.interceptors.request.use(
    (config: InternalAxiosRequestConfig) => {
      // Attach auth token
      const token = storage.getString('auth_token');
      if (token) {
        config.headers.Authorization = `Bearer ${token}`;
      }

      // Attach device ID for audit trail
      const deviceId = storage.getString('device_id');
      if (deviceId) {
        config.headers['X-Device-Id'] = deviceId;
      }

      // Timestamp for latency tracking
      (config as any).__startTime = Date.now();

      return config;
    },
    (error) => Promise.reject(error)
  );

  // --- RESPONSE INTERCEPTOR ---
  client.interceptors.response.use(
    (response) => {
      // Log latency in dev mode
      const startTime = (response.config as any).__startTime;
      if (startTime && __DEV__) {
        const latency = Date.now() - startTime;
        console.log(`[API] ${response.config.url} — ${latency}ms`);
      }
      return response;
    },
    async (error: AxiosError) => {
      const config = error.config as any;

      // Retry logic for network errors and 5xx responses
      if (
        config &&
        !config.__retryCount &&
        (error.code === 'ECONNABORTED' ||
          error.code === 'ERR_NETWORK' ||
          (error.response && error.response.status >= 500))
      ) {
        config.__retryCount = (config.__retryCount || 0) + 1;
        if (config.__retryCount <= API_CONFIG.RETRY_COUNT) {
          await new Promise((r) => setTimeout(r, API_CONFIG.RETRY_DELAY));
          return client(config);
        }
      }

      return Promise.reject(error);
    }
  );

  return client;
}

export const apiClient = createApiClient();
```

### Face Detection API

```typescript
// services/faceApi.ts
import { apiClient } from './api';

// ------- REQUEST / RESPONSE TYPES -------

export interface DetectFaceRequest {
  image_base64: string;          // Base64-encoded JPEG of cropped face
  device_id: string;             // Mobile device identifier
  location_id?: string;          // Physical location (e.g., office, gate)
  timestamp: string;             // ISO 8601 capture time
}

export interface DetectFaceResponse {
  success: boolean;
  person: {
    id: string;                  // Employee/student ID
    name: string;                // Full name
    department: string;          // Department / class
    avatar_url?: string;         // Profile photo URL
  } | null;
  confidence: number;            // 0.0 - 1.0 match confidence
  embedding_time_ms: number;     // Server embedding generation time
  search_time_ms: number;        // Server FAISS search time
  total_time_ms: number;         // Total server processing time
  attendance: {
    status: 'checked_in' | 'already_checked_in' | 'unknown';
    check_in_time?: string;      // ISO 8601
    is_late: boolean;
  };
}

export interface HealthCheckResponse {
  status: 'ok' | 'degraded' | 'down';
  gpu_available: boolean;
  faiss_index_size: number;      // Number of registered faces
  uptime_seconds: number;
  queue_depth: number;           // Pending inference requests
  avg_latency_ms: number;
}

// ------- API FUNCTIONS -------

/**
 * Send a cropped face image to the server for recognition.
 *
 * Endpoint: POST /api/v1/detect
 * Expected response time: 15-25ms (local network)
 */
export async function detectFace(
  request: DetectFaceRequest
): Promise<DetectFaceResponse> {
  const response = await apiClient.post<DetectFaceResponse>(
    '/api/v1/detect',
    request
  );
  return response.data;
}

/**
 * Register a new face for a person (enrollment).
 *
 * Endpoint: POST /api/v1/register
 * Should capture 3-5 images from different angles.
 */
export async function registerFace(
  personId: string,
  images: string[],  // Array of base64 face images
  metadata: { name: string; department: string }
): Promise<{ success: boolean; embedding_count: number }> {
  const response = await apiClient.post('/api/v1/register', {
    person_id: personId,
    images,
    metadata,
  });
  return response.data;
}

/**
 * Health check — verify server is reachable and GPU is working.
 *
 * Endpoint: GET /api/v1/health
 * Call this on app startup and periodically (every 30s).
 */
export async function healthCheck(): Promise<HealthCheckResponse> {
  const response = await apiClient.get<HealthCheckResponse>(
    '/api/v1/health',
    { timeout: 3000 }  // Short timeout for health checks
  );
  return response.data;
}

/**
 * Get today's attendance records for a location.
 *
 * Endpoint: GET /api/v1/attendance
 */
export async function getAttendance(
  locationId: string,
  date: string  // YYYY-MM-DD
): Promise<{
  records: Array<{
    person_id: string;
    name: string;
    check_in_time: string;
    is_late: boolean;
  }>;
  total: number;
  present: number;
  late: number;
}> {
  const response = await apiClient.get('/api/v1/attendance', {
    params: { location_id: locationId, date },
  });
  return response.data;
}
```

---

## 8. UI/UX — Face Capture Screen

### Main Screen

```typescript
// screens/FaceCaptureScreen.tsx
import React, { useCallback, useEffect, useRef, useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  Dimensions,
  StatusBar,
  Animated,
  Platform,
} from 'react-native';
import HapticFeedback from 'react-native-haptic-feedback';
import { CameraPreview } from '../components/CameraPreview';
import { FaceGuideFrame } from '../components/FaceGuideFrame';
import { ResultCard } from '../components/ResultCard';
import { AttendanceStatusBar } from '../components/AttendanceStatusBar';
import { CaptureButton } from '../components/CaptureButton';
import { useFaceDetection } from '../hooks/useFaceDetection';
import { useAttendance } from '../hooks/useAttendance';

const { width: SCREEN_W, height: SCREEN_H } = Dimensions.get('window');

// Color palette matching the presentation dark theme
const COLORS = {
  bg: '#0B1120',
  card: '#131D33',
  accent: '#00D4AA',
  accentBlue: '#0EA5E9',
  accentPurple: '#8B5CF6',
  white: '#FFFFFF',
  muted: '#7B8FA3',
  text: '#CBD5E1',
  success: '#10B981',
  warning: '#F59E0B',
  error: '#FF6B6B',
};

type CaptureState =
  | 'idle'             // Waiting for face
  | 'detecting'        // Face found, validating
  | 'ready'            // Face validated, ready to capture
  | 'capturing'        // Taking photo
  | 'processing'       // Sending to server
  | 'success'          // Identity matched
  | 'unknown'          // No match found
  | 'error';           // Network or server error

export function FaceCaptureScreen() {
  const [captureState, setCaptureState] = useState<CaptureState>('idle');
  const [serverHealth, setServerHealth] = useState<'ok' | 'degraded' | 'down'>('ok');
  const pulseAnim = useRef(new Animated.Value(1)).current;

  const { detectedFaces, validation, onFaceDetectionResult } = useFaceDetection();
  const {
    result,
    isProcessing,
    captureAndRecognize,
    clearResult,
    latencyMs,
  } = useAttendance();

  // --- AUTO-CAPTURE LOGIC ---
  // When face is valid for 1.5 seconds, auto-trigger capture
  const validSinceRef = useRef<number | null>(null);
  const AUTO_CAPTURE_DELAY = 1500; // ms

  useEffect(() => {
    if (validation.isValid && captureState === 'detecting') {
      if (!validSinceRef.current) {
        validSinceRef.current = Date.now();
      }

      const elapsed = Date.now() - validSinceRef.current;
      if (elapsed >= AUTO_CAPTURE_DELAY) {
        handleCapture();
        validSinceRef.current = null;
      }
    } else {
      validSinceRef.current = null;
    }
  }, [validation.isValid, captureState]);

  // --- STATE TRANSITIONS ---
  useEffect(() => {
    if (detectedFaces.length === 1 && captureState === 'idle') {
      setCaptureState('detecting');
    }
    if (detectedFaces.length === 0 && captureState === 'detecting') {
      setCaptureState('idle');
    }
    if (validation.isValid && captureState === 'detecting') {
      setCaptureState('ready');
    }
    if (!validation.isValid && captureState === 'ready') {
      setCaptureState('detecting');
    }
  }, [detectedFaces, validation, captureState]);

  // --- PULSE ANIMATION (when ready) ---
  useEffect(() => {
    if (captureState === 'ready') {
      Animated.loop(
        Animated.sequence([
          Animated.timing(pulseAnim, {
            toValue: 1.15,
            duration: 600,
            useNativeDriver: true,
          }),
          Animated.timing(pulseAnim, {
            toValue: 1,
            duration: 600,
            useNativeDriver: true,
          }),
        ])
      ).start();
    } else {
      pulseAnim.setValue(1);
    }
  }, [captureState]);

  // --- CAPTURE HANDLER ---
  const handleCapture = useCallback(async () => {
    if (captureState !== 'ready' || isProcessing) return;

    setCaptureState('capturing');
    HapticFeedback.trigger('impactMedium');

    try {
      setCaptureState('processing');
      const res = await captureAndRecognize(detectedFaces[0]);

      if (res.success && res.person) {
        setCaptureState('success');
        HapticFeedback.trigger('notificationSuccess');
      } else {
        setCaptureState('unknown');
        HapticFeedback.trigger('notificationWarning');
      }

      // Reset after 3 seconds
      setTimeout(() => {
        setCaptureState('idle');
        clearResult();
      }, 3000);
    } catch (err) {
      setCaptureState('error');
      HapticFeedback.trigger('notificationError');
      setTimeout(() => setCaptureState('idle'), 2000);
    }
  }, [captureState, isProcessing, detectedFaces, captureAndRecognize, clearResult]);

  // --- STATUS TEXT ---
  const getStatusText = (): string => {
    switch (captureState) {
      case 'idle': return 'Position your face in the frame';
      case 'detecting':
        return validation.issues[0] || 'Adjusting...';
      case 'ready': return 'Hold still — capturing...';
      case 'capturing': return 'Capturing...';
      case 'processing': return 'Recognizing...';
      case 'success': return `Welcome, ${result?.person?.name}!`;
      case 'unknown': return 'Face not recognized';
      case 'error': return 'Connection error — retrying...';
    }
  };

  const getStatusColor = (): string => {
    switch (captureState) {
      case 'idle': return COLORS.muted;
      case 'detecting': return COLORS.warning;
      case 'ready': return COLORS.accent;
      case 'capturing':
      case 'processing': return COLORS.accentBlue;
      case 'success': return COLORS.success;
      case 'unknown': return COLORS.warning;
      case 'error': return COLORS.error;
    }
  };

  return (
    <View style={styles.container}>
      <StatusBar barStyle="light-content" backgroundColor={COLORS.bg} />

      {/* Server health indicator */}
      <AttendanceStatusBar
        health={serverHealth}
        captureState={captureState}
        latencyMs={latencyMs}
      />

      {/* Camera + face detection */}
      <View style={styles.cameraContainer}>
        <CameraPreview
          onFaceDetected={onFaceDetectionResult}
          isProcessing={isProcessing}
        />

        {/* Animated face guide oval */}
        <FaceGuideFrame
          quality={validation.quality}
          isCapturing={captureState === 'capturing' || captureState === 'processing'}
        />

        {/* Bounding box overlay */}
        {detectedFaces.length > 0 && captureState !== 'success' && (
          <View style={[
            styles.boundingBox,
            {
              left: `${detectedFaces[0].bounds.x * 100}%`,
              top: `${detectedFaces[0].bounds.y * 100}%`,
              width: `${detectedFaces[0].bounds.width * 100}%`,
              height: `${detectedFaces[0].bounds.height * 100}%`,
              borderColor: getStatusColor(),
            },
          ]} />
        )}
      </View>

      {/* Status text */}
      <View style={styles.statusContainer}>
        <Text style={[styles.statusText, { color: getStatusColor() }]}>
          {getStatusText()}
        </Text>

        {/* Latency indicator */}
        {latencyMs > 0 && captureState === 'success' && (
          <Text style={styles.latencyText}>{latencyMs}ms round-trip</Text>
        )}
      </View>

      {/* Result card (slides up on success/unknown) */}
      {(captureState === 'success' || captureState === 'unknown') && result && (
        <ResultCard
          result={result}
          onDismiss={() => {
            setCaptureState('idle');
            clearResult();
          }}
        />
      )}

      {/* Manual capture button (fallback) */}
      <Animated.View style={[
        styles.captureButtonContainer,
        { transform: [{ scale: pulseAnim }] },
      ]}>
        <CaptureButton
          state={captureState}
          onPress={handleCapture}
          disabled={!validation.isValid || isProcessing}
        />
      </Animated.View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: COLORS.bg,
  },
  cameraContainer: {
    flex: 1,
    overflow: 'hidden',
    borderRadius: 16,
    marginHorizontal: 16,
    marginTop: 8,
  },
  boundingBox: {
    position: 'absolute',
    borderWidth: 2,
    borderRadius: 8,
    borderStyle: 'dashed',
  },
  statusContainer: {
    alignItems: 'center',
    paddingVertical: 16,
  },
  statusText: {
    fontSize: 16,
    fontWeight: '600',
    letterSpacing: 0.3,
  },
  latencyText: {
    fontSize: 12,
    color: COLORS.muted,
    marginTop: 4,
  },
  captureButtonContainer: {
    alignItems: 'center',
    paddingBottom: Platform.OS === 'ios' ? 40 : 24,
  },
});
```

### Face Guide Frame Component

```typescript
// components/FaceGuideFrame.tsx
import React, { useEffect, useRef } from 'react';
import { View, StyleSheet, Animated, Dimensions } from 'react-native';
import Svg, { Ellipse, Defs, Mask, Rect } from 'react-native-svg';

const { width: W } = Dimensions.get('window');
const OVAL_W = W * 0.55;
const OVAL_H = OVAL_W * 1.35;

interface FaceGuideFrameProps {
  quality: 'good' | 'fair' | 'poor';
  isCapturing: boolean;
}

export function FaceGuideFrame({ quality, isCapturing }: FaceGuideFrameProps) {
  const borderColor =
    quality === 'good' ? '#00D4AA' :
    quality === 'fair' ? '#F59E0B' : '#7B8FA3';

  const opacity = useRef(new Animated.Value(0.5)).current;

  useEffect(() => {
    if (isCapturing) {
      // Flash animation during capture
      Animated.sequence([
        Animated.timing(opacity, { toValue: 1, duration: 100, useNativeDriver: true }),
        Animated.timing(opacity, { toValue: 0.5, duration: 300, useNativeDriver: true }),
      ]).start();
    }
  }, [isCapturing]);

  return (
    <View style={StyleSheet.absoluteFill} pointerEvents="none">
      {/* Semi-transparent mask outside the oval */}
      <Svg style={StyleSheet.absoluteFill}>
        <Defs>
          <Mask id="mask">
            <Rect width="100%" height="100%" fill="white" />
            <Ellipse
              cx="50%"
              cy="45%"
              rx={OVAL_W / 2}
              ry={OVAL_H / 2}
              fill="black"
            />
          </Mask>
        </Defs>
        <Rect
          width="100%"
          height="100%"
          fill="rgba(11,17,32,0.65)"
          mask="url(#mask)"
        />
        {/* Guide oval border */}
        <Ellipse
          cx="50%"
          cy="45%"
          rx={OVAL_W / 2}
          ry={OVAL_H / 2}
          fill="none"
          stroke={borderColor}
          strokeWidth={3}
          strokeDasharray={quality === 'poor' ? '10,8' : '0'}
        />
      </Svg>

      {/* Corner brackets for visual flair */}
      {quality === 'good' && (
        <>
          <View style={[styles.corner, styles.topLeft, { borderColor }]} />
          <View style={[styles.corner, styles.topRight, { borderColor }]} />
          <View style={[styles.corner, styles.bottomLeft, { borderColor }]} />
          <View style={[styles.corner, styles.bottomRight, { borderColor }]} />
        </>
      )}
    </View>
  );
}

const CORNER_SIZE = 24;
const styles = StyleSheet.create({
  corner: {
    position: 'absolute',
    width: CORNER_SIZE,
    height: CORNER_SIZE,
    borderWidth: 3,
  },
  topLeft: {
    top: '20%', left: '18%',
    borderRightWidth: 0, borderBottomWidth: 0,
    borderTopLeftRadius: 8,
  },
  topRight: {
    top: '20%', right: '18%',
    borderLeftWidth: 0, borderBottomWidth: 0,
    borderTopRightRadius: 8,
  },
  bottomLeft: {
    bottom: '30%', left: '18%',
    borderRightWidth: 0, borderTopWidth: 0,
    borderBottomLeftRadius: 8,
  },
  bottomRight: {
    bottom: '30%', right: '18%',
    borderLeftWidth: 0, borderTopWidth: 0,
    borderBottomRightRadius: 8,
  },
});
```

### Result Card Component

```typescript
// components/ResultCard.tsx
import React, { useEffect, useRef } from 'react';
import { View, Text, StyleSheet, Animated, Image } from 'react-native';
import { DetectFaceResponse } from '../services/faceApi';

interface ResultCardProps {
  result: DetectFaceResponse;
  onDismiss: () => void;
}

export function ResultCard({ result, onDismiss }: ResultCardProps) {
  const slideAnim = useRef(new Animated.Value(200)).current;
  const opacityAnim = useRef(new Animated.Value(0)).current;

  useEffect(() => {
    Animated.parallel([
      Animated.spring(slideAnim, {
        toValue: 0,
        tension: 100,
        friction: 12,
        useNativeDriver: true,
      }),
      Animated.timing(opacityAnim, {
        toValue: 1,
        duration: 200,
        useNativeDriver: true,
      }),
    ]).start();
  }, []);

  const isMatch = result.success && result.person;
  const confidence = Math.round(result.confidence * 100);

  return (
    <Animated.View style={[
      styles.card,
      {
        transform: [{ translateY: slideAnim }],
        opacity: opacityAnim,
        borderLeftColor: isMatch ? '#10B981' : '#F59E0B',
      },
    ]}>
      {isMatch ? (
        <View style={styles.row}>
          {result.person?.avatar_url && (
            <Image
              source={{ uri: result.person.avatar_url }}
              style={styles.avatar}
            />
          )}
          <View style={styles.info}>
            <Text style={styles.name}>{result.person?.name}</Text>
            <Text style={styles.dept}>{result.person?.department}</Text>
            <View style={styles.metaRow}>
              <Text style={styles.confidence}>{confidence}% match</Text>
              <Text style={styles.timing}>
                {result.total_time_ms}ms server
              </Text>
            </View>
          </View>
          <View style={[
            styles.statusBadge,
            {
              backgroundColor: result.attendance.is_late
                ? 'rgba(245,158,11,0.15)'
                : 'rgba(16,185,129,0.15)',
            },
          ]}>
            <Text style={[
              styles.statusText,
              {
                color: result.attendance.is_late ? '#F59E0B' : '#10B981',
              },
            ]}>
              {result.attendance.is_late ? 'LATE' : 'ON TIME'}
            </Text>
          </View>
        </View>
      ) : (
        <View style={styles.row}>
          <View style={styles.unknownIcon}>
            <Text style={styles.unknownEmoji}>?</Text>
          </View>
          <View style={styles.info}>
            <Text style={styles.name}>Unknown Person</Text>
            <Text style={styles.dept}>Face not registered in the system</Text>
          </View>
        </View>
      )}
    </Animated.View>
  );
}

const styles = StyleSheet.create({
  card: {
    position: 'absolute',
    bottom: 120,
    left: 20,
    right: 20,
    backgroundColor: '#131D33',
    borderRadius: 16,
    padding: 16,
    borderLeftWidth: 4,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 8 },
    shadowOpacity: 0.3,
    shadowRadius: 16,
    elevation: 10,
  },
  row: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  avatar: {
    width: 52,
    height: 52,
    borderRadius: 26,
    marginRight: 12,
  },
  info: {
    flex: 1,
  },
  name: {
    color: '#FFFFFF',
    fontSize: 18,
    fontWeight: '700',
  },
  dept: {
    color: '#7B8FA3',
    fontSize: 13,
    marginTop: 2,
  },
  metaRow: {
    flexDirection: 'row',
    gap: 12,
    marginTop: 4,
  },
  confidence: {
    color: '#00D4AA',
    fontSize: 12,
    fontWeight: '600',
  },
  timing: {
    color: '#7B8FA3',
    fontSize: 12,
  },
  statusBadge: {
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 8,
  },
  statusText: {
    fontSize: 11,
    fontWeight: '800',
    letterSpacing: 1,
  },
  unknownIcon: {
    width: 52,
    height: 52,
    borderRadius: 26,
    backgroundColor: 'rgba(245,158,11,0.15)',
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: 12,
  },
  unknownEmoji: {
    fontSize: 24,
    color: '#F59E0B',
    fontWeight: '700',
  },
});
```

---

## 9. State Management

```typescript
// store/attendanceStore.ts
import { create } from 'zustand';
import { DetectFaceResponse } from '../services/faceApi';

interface AttendanceRecord {
  personId: string;
  name: string;
  department: string;
  checkInTime: string;
  confidence: number;
  latencyMs: number;
  isLate: boolean;
}

interface AttendanceState {
  // Session state
  todayRecords: AttendanceRecord[];
  totalCheckedIn: number;
  totalLate: number;
  locationId: string;

  // Server health
  serverStatus: 'ok' | 'degraded' | 'down' | 'unknown';
  lastHealthCheck: number;
  avgLatencyMs: number;

  // Capture state
  lastResult: DetectFaceResponse | null;
  isProcessing: boolean;
  consecutiveErrors: number;

  // Actions
  addRecord: (record: AttendanceRecord) => void;
  setServerStatus: (status: 'ok' | 'degraded' | 'down' | 'unknown') => void;
  setLastResult: (result: DetectFaceResponse | null) => void;
  setProcessing: (processing: boolean) => void;
  incrementErrors: () => void;
  resetErrors: () => void;
  reset: () => void;
}

export const useAttendanceStore = create<AttendanceState>((set, get) => ({
  todayRecords: [],
  totalCheckedIn: 0,
  totalLate: 0,
  locationId: 'main-entrance',

  serverStatus: 'unknown',
  lastHealthCheck: 0,
  avgLatencyMs: 0,

  lastResult: null,
  isProcessing: false,
  consecutiveErrors: 0,

  addRecord: (record) =>
    set((state) => ({
      todayRecords: [record, ...state.todayRecords],
      totalCheckedIn: state.totalCheckedIn + 1,
      totalLate: state.totalLate + (record.isLate ? 1 : 0),
    })),

  setServerStatus: (status) =>
    set({ serverStatus: status, lastHealthCheck: Date.now() }),

  setLastResult: (result) => set({ lastResult: result }),
  setProcessing: (processing) => set({ isProcessing: processing }),

  incrementErrors: () =>
    set((state) => ({ consecutiveErrors: state.consecutiveErrors + 1 })),

  resetErrors: () => set({ consecutiveErrors: 0 }),

  reset: () =>
    set({
      todayRecords: [],
      totalCheckedIn: 0,
      totalLate: 0,
      lastResult: null,
      isProcessing: false,
      consecutiveErrors: 0,
    }),
}));
```

---

## 10. Attendance Flow & Business Logic

```typescript
// hooks/useAttendance.ts
import { useCallback, useRef, useState, useEffect } from 'react';
import { useCamera } from './useCamera';
import { cropFaceFromPhoto, checkImageQuality } from '../utils/imageProcessing';
import { detectFace, healthCheck, DetectFaceResponse } from '../services/faceApi';
import { useAttendanceStore } from '../store/attendanceStore';
import { MMKV } from 'react-native-mmkv';

const storage = new MMKV();
const HEALTH_CHECK_INTERVAL = 30_000; // 30 seconds

export function useAttendance() {
  const { takeSnapshot } = useCamera();
  const [latencyMs, setLatencyMs] = useState(0);
  const healthIntervalRef = useRef<NodeJS.Timeout | null>(null);

  const {
    lastResult,
    isProcessing,
    setLastResult,
    setProcessing,
    setServerStatus,
    addRecord,
    incrementErrors,
    resetErrors,
    locationId,
  } = useAttendanceStore();

  // --- HEALTH CHECK LOOP ---
  useEffect(() => {
    const checkHealth = async () => {
      try {
        const health = await healthCheck();
        setServerStatus(health.status);
      } catch {
        setServerStatus('down');
      }
    };

    checkHealth(); // Initial check
    healthIntervalRef.current = setInterval(checkHealth, HEALTH_CHECK_INTERVAL);
    return () => {
      if (healthIntervalRef.current) clearInterval(healthIntervalRef.current);
    };
  }, []);

  // --- MAIN CAPTURE + RECOGNIZE FLOW ---
  const captureAndRecognize = useCallback(
    async (detectedFace: { bounds: any }): Promise<DetectFaceResponse> => {
      if (isProcessing) throw new Error('Already processing');

      setProcessing(true);
      const startTime = Date.now();

      try {
        // Step 1: Take photo snapshot
        const photo = await takeSnapshot();
        if (!photo) throw new Error('Failed to capture photo');

        // Step 2: Crop face region
        const cropped = await cropFaceFromPhoto(
          photo.path,
          detectedFace.bounds
        );

        // Step 3: Quality check
        const quality = checkImageQuality(cropped.base64);
        if (!quality.isAcceptable) {
          throw new Error(
            quality.brightness !== 'ok'
              ? `Image too ${quality.brightness === 'too_dark' ? 'dark' : 'bright'}`
              : 'Image is blurry'
          );
        }

        // Step 4: Send to server
        const deviceId = storage.getString('device_id') || 'unknown';
        const result = await detectFace({
          image_base64: cropped.base64,
          device_id: deviceId,
          location_id: locationId,
          timestamp: new Date().toISOString(),
        });

        // Step 5: Update state
        const totalLatency = Date.now() - startTime;
        setLatencyMs(totalLatency);
        setLastResult(result);
        resetErrors();

        // Step 6: Record attendance if matched
        if (result.success && result.person) {
          addRecord({
            personId: result.person.id,
            name: result.person.name,
            department: result.person.department,
            checkInTime: result.attendance.check_in_time || new Date().toISOString(),
            confidence: result.confidence,
            latencyMs: totalLatency,
            isLate: result.attendance.is_late,
          });
        }

        return result;
      } catch (error) {
        incrementErrors();
        throw error;
      } finally {
        setProcessing(false);
      }
    },
    [isProcessing, takeSnapshot, locationId]
  );

  const clearResult = useCallback(() => {
    setLastResult(null);
    setLatencyMs(0);
  }, []);

  return {
    result: lastResult,
    isProcessing,
    captureAndRecognize,
    clearResult,
    latencyMs,
  };
}
```

---

## 11. Error Handling & Edge Cases

### Offline Queue (for unreliable networks)

```typescript
// services/offlineQueue.ts
import { MMKV } from 'react-native-mmkv';
import NetInfo from '@react-native-community/netinfo';
import { detectFace, DetectFaceRequest } from './faceApi';

const storage = new MMKV();
const QUEUE_KEY = 'offline_queue';
const MAX_QUEUE_SIZE = 50;

interface QueuedRequest {
  id: string;
  request: DetectFaceRequest;
  timestamp: number;
  retries: number;
}

/**
 * Queue a face detection request for when network is available.
 * Used for spotty network conditions (e.g., outdoor deployments).
 */
export function enqueueRequest(request: DetectFaceRequest): void {
  const queue = getQueue();
  if (queue.length >= MAX_QUEUE_SIZE) {
    queue.shift(); // Drop oldest
  }

  queue.push({
    id: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
    request,
    timestamp: Date.now(),
    retries: 0,
  });

  storage.set(QUEUE_KEY, JSON.stringify(queue));
}

/**
 * Process all queued requests.
 * Called when network connectivity is restored.
 */
export async function processQueue(): Promise<number> {
  const queue = getQueue();
  if (queue.length === 0) return 0;

  let processed = 0;
  const remaining: QueuedRequest[] = [];

  for (const item of queue) {
    // Skip requests older than 1 hour
    if (Date.now() - item.timestamp > 3_600_000) continue;

    try {
      await detectFace(item.request);
      processed++;
    } catch {
      item.retries++;
      if (item.retries < 3) {
        remaining.push(item);
      }
    }
  }

  storage.set(QUEUE_KEY, JSON.stringify(remaining));
  return processed;
}

function getQueue(): QueuedRequest[] {
  const raw = storage.getString(QUEUE_KEY);
  return raw ? JSON.parse(raw) : [];
}

// --- AUTO-PROCESS ON NETWORK CHANGE ---
export function startQueueWatcher(): () => void {
  const unsubscribe = NetInfo.addEventListener(async (state) => {
    if (state.isConnected && state.isInternetReachable) {
      const count = await processQueue();
      if (count > 0) {
        console.log(`[OfflineQueue] Processed ${count} queued requests`);
      }
    }
  });
  return unsubscribe;
}
```

### Error Handling Matrix

| Error Type | Detection | Response | UX |
|------------|-----------|----------|----|
| No face detected | `faces.length === 0` | Show guide text | "Position your face in the frame" |
| Multiple faces | `faces.length > 1` | Block capture | "Only one face allowed" |
| Face too small | `bounds.width < 0.25` | Show guide text | "Move closer" |
| Face angle bad | `yawAngle > 15°` | Show guide text | "Face the camera directly" |
| Image too dark | Brightness check | Flash hint / retry | "Improve lighting" |
| Image blurry | Variance check | Retry auto-capture | "Hold still" |
| Network timeout | `ECONNABORTED` | Auto-retry x2, then queue | "Retrying..." |
| Server 500 | HTTP status | Auto-retry x2, then error card | "Server error" |
| Server down | Health check fails | Show offline badge | "Server offline — queuing" |
| No match found | `confidence < threshold` | Show unknown card | "Face not recognized" |
| Already checked in | Server response | Show info card | "Already checked in at 8:42 AM" |
| Camera permission denied | Permission API | Show settings prompt | "Camera access needed" |

---

## 12. Performance Optimization

### Key Optimizations Checklist

```typescript
// utils/constants.ts — Performance tuning

export const PERF = {
  // Camera
  CAMERA_RESOLUTION: { width: 640, height: 480 },  // NOT full HD
  TARGET_FPS: 30,
  PHOTO_QUALITY: 'speed',  // Not 'quality'

  // Face detection
  DETECTION_THROTTLE_MS: 40,     // Max ~25 detections/sec
  MIN_FACE_SIZE: 0.25,           // Skip tiny faces
  PERFORMANCE_MODE: 'fast',      // Not 'accurate'

  // Image processing
  CROP_TARGET_SIZE: 224,          // 224x224 px — standard model input
  JPEG_QUALITY: 85,               // Balance quality vs size
  MAX_PAYLOAD_KB: 100,            // Reject if > 100KB

  // Network
  API_TIMEOUT_MS: 10_000,        // 10 seconds
  RETRY_COUNT: 2,
  RETRY_DELAY_MS: 500,
  HEALTH_CHECK_INTERVAL_MS: 30_000,

  // Auto-capture
  AUTO_CAPTURE_DELAY_MS: 1500,   // Face must be valid for 1.5s
  RESULT_DISPLAY_MS: 3000,       // Show result for 3 seconds
  COOLDOWN_BETWEEN_CAPTURES_MS: 2000,

  // Anti-spam
  MAX_CAPTURES_PER_MINUTE: 20,   // Rate limit
  DUPLICATE_CHECK_WINDOW_MS: 5000, // Don't re-capture same face within 5s
};
```

### Memory Management

```typescript
// Cleanup camera resources when screen is not visible
useEffect(() => {
  const subscription = AppState.addEventListener('change', (state) => {
    if (state === 'background') {
      setIsActive(false);      // Pause camera
    } else if (state === 'active') {
      setIsActive(true);       // Resume camera
    }
  });

  return () => subscription.remove();
}, []);

// Delete temporary crop files after processing
async function cleanupTempFiles() {
  const tempDir = RNFS.CachesDirectoryPath + '/face_crops';
  const files = await RNFS.readDir(tempDir);
  const oldFiles = files.filter(
    (f) => Date.now() - f.mtime.getTime() > 60_000  // Older than 1 min
  );
  for (const file of oldFiles) {
    await RNFS.unlink(file.path);
  }
}
```

---

## 13. Security Considerations

### Transport Security

```typescript
// Enforce HTTPS in production + certificate pinning
import { Platform } from 'react-native';

const SECURITY_CONFIG = {
  // Certificate pinning — pin the leaf cert SHA256
  pinnedCerts: [
    'sha256/XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX=',
  ],

  // Disable in dev for local Jetson testing
  enforcePinning: !__DEV__,

  // Token refresh interval
  tokenRefreshIntervalMs: 3_600_000, // 1 hour
};
```

### Anti-Spoofing

```typescript
/**
 * Basic liveness checks (client-side).
 * These complement server-side anti-spoofing in AntelopeV2.
 *
 * For production, use a dedicated liveness SDK:
 * - FaceTec (most robust)
 * - iProov
 * - Onfido
 */
export function livenessCheck(face: DetectedFace, frames: DetectedFace[]): {
  isLive: boolean;
  reason?: string;
} {
  // Check 1: Eye blink detected in recent frames
  const blinksDetected = frames.some(
    (f) => f.leftEyeOpenProbability < 0.3 || f.rightEyeOpenProbability < 0.3
  );

  // Check 2: Head moved slightly (not a static photo)
  const headMovement = frames.length > 5 && (
    Math.abs(frames[0].yawAngle - frames[frames.length - 1].yawAngle) > 2 ||
    Math.abs(frames[0].pitchAngle - frames[frames.length - 1].pitchAngle) > 2
  );

  // Check 3: Face size changed (depth)
  const sizeVariance = frames.length > 5 &&
    Math.abs(frames[0].bounds.width - frames[frames.length - 1].bounds.width) > 0.01;

  if (!blinksDetected && !headMovement) {
    return { isLive: false, reason: 'No movement detected — possible photo attack' };
  }

  return { isLive: true };
}
```

### Data Privacy

```typescript
/**
 * IMPORTANT: Face data handling policy
 *
 * 1. NEVER store face images on device longer than the capture session
 * 2. Base64 face data is held in memory only during API call
 * 3. Temp crop files are deleted immediately after upload
 * 4. Face embeddings are stored server-side only (never on device)
 * 5. Device stores only: person ID, name, check-in time (no biometrics)
 * 6. All local attendance records are encrypted via MMKV
 * 7. Comply with local biometric data laws (BIPA, GDPR Art. 9, etc.)
 */
```

---

## 14. Testing Strategy

### Unit Tests

```typescript
// __tests__/useFaceDetection.test.ts
import { renderHook, act } from '@testing-library/react-hooks';
import { useFaceDetection } from '../hooks/useFaceDetection';

describe('useFaceDetection', () => {
  it('validates a good face', () => {
    const { result } = renderHook(() => useFaceDetection());

    act(() => {
      result.current.onFaceDetectionResult([{
        bounds: { x: 0.3, y: 0.25, width: 0.4, height: 0.5 },
        trackingId: 1,
        rightEyeOpenProbability: 0.95,
        leftEyeOpenProbability: 0.92,
        smilingProbability: 0.3,
        rollAngle: 2,
        yawAngle: -3,
        pitchAngle: 1,
      }]);
    });

    expect(result.current.validation.isValid).toBe(true);
    expect(result.current.validation.quality).toBe('good');
  });

  it('rejects face that is too small', () => {
    const { result } = renderHook(() => useFaceDetection());

    act(() => {
      result.current.onFaceDetectionResult([{
        bounds: { x: 0.4, y: 0.4, width: 0.1, height: 0.12 },
        trackingId: 1,
        rightEyeOpenProbability: 0.9,
        leftEyeOpenProbability: 0.9,
        smilingProbability: 0.5,
        rollAngle: 0,
        yawAngle: 0,
        pitchAngle: 0,
      }]);
    });

    expect(result.current.validation.isValid).toBe(false);
    expect(result.current.validation.issues).toContain('Move closer — face too small');
  });

  it('rejects multiple faces', () => {
    const { result } = renderHook(() => useFaceDetection());

    act(() => {
      result.current.onFaceDetectionResult([
        { bounds: { x: 0.1, y: 0.2, width: 0.3, height: 0.4 }, trackingId: 1 } as any,
        { bounds: { x: 0.5, y: 0.2, width: 0.3, height: 0.4 }, trackingId: 2 } as any,
      ]);
    });

    expect(result.current.validation.isValid).toBe(false);
    expect(result.current.validation.issues).toContain(
      'Multiple faces detected — only one allowed'
    );
  });

  it('rejects turned face', () => {
    const { result } = renderHook(() => useFaceDetection());

    act(() => {
      result.current.onFaceDetectionResult([{
        bounds: { x: 0.3, y: 0.25, width: 0.4, height: 0.5 },
        trackingId: 1,
        rightEyeOpenProbability: 0.9,
        leftEyeOpenProbability: 0.9,
        smilingProbability: 0.3,
        rollAngle: 0,
        yawAngle: 25,   // Turned too far
        pitchAngle: 0,
      }]);
    });

    expect(result.current.validation.isValid).toBe(false);
    expect(result.current.validation.issues).toContain('Face the camera directly');
  });
});
```

### API Mock for Testing

```typescript
// __mocks__/faceApi.ts
import { DetectFaceResponse } from '../services/faceApi';

export const mockSuccessResponse: DetectFaceResponse = {
  success: true,
  person: {
    id: 'EMP001',
    name: 'Ravi Kumar',
    department: 'Engineering',
    avatar_url: 'https://api.example.com/avatars/EMP001.jpg',
  },
  confidence: 0.97,
  embedding_time_ms: 12,
  search_time_ms: 1,
  total_time_ms: 15,
  attendance: {
    status: 'checked_in',
    check_in_time: '2025-03-12T08:45:00Z',
    is_late: false,
  },
};

export const mockUnknownResponse: DetectFaceResponse = {
  success: false,
  person: null,
  confidence: 0.23,
  embedding_time_ms: 11,
  search_time_ms: 2,
  total_time_ms: 15,
  attendance: {
    status: 'unknown',
    is_late: false,
  },
};
```

---

## 15. Full Code Implementation

### Server-Side Endpoint (FastAPI — for reference)

```python
# server/main.py — FastAPI endpoint that the mobile app calls

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import faiss
import base64
import cv2
from insightface.app import FaceAnalysis
import time

app = FastAPI(title="Face Recognition API")

# Initialize model (runs once on startup)
face_app = FaceAnalysis(
    name="antelopev2",
    root="./models",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
)
face_app.prepare(ctx_id=0, det_size=(160, 160))

# Load FAISS index
index = faiss.read_index("./data/faces.index")
person_lookup: dict[int, dict] = load_person_database()  # ID → {name, dept}

CONFIDENCE_THRESHOLD = 0.45  # cosine distance threshold


class DetectRequest(BaseModel):
    image_base64: str
    device_id: str
    location_id: str | None = None
    timestamp: str


class DetectResponse(BaseModel):
    success: bool
    person: dict | None
    confidence: float
    embedding_time_ms: float
    search_time_ms: float
    total_time_ms: float
    attendance: dict


@app.post("/api/v1/detect", response_model=DetectResponse)
async def detect_face(request: DetectRequest):
    total_start = time.perf_counter()

    # Decode base64 image
    try:
        img_bytes = base64.b64decode(request.image_base64)
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    except Exception:
        raise HTTPException(400, "Invalid image data")

    if img is None:
        raise HTTPException(400, "Could not decode image")

    # Generate embedding
    embed_start = time.perf_counter()
    faces = face_app.get(img)
    embed_ms = (time.perf_counter() - embed_start) * 1000

    if len(faces) == 0:
        return DetectResponse(
            success=False,
            person=None,
            confidence=0.0,
            embedding_time_ms=round(embed_ms, 1),
            search_time_ms=0,
            total_time_ms=round((time.perf_counter() - total_start) * 1000, 1),
            attendance={"status": "unknown", "is_late": False},
        )

    embedding = faces[0].normed_embedding.reshape(1, -1).astype("float32")

    # FAISS search
    search_start = time.perf_counter()
    distances, indices = index.search(embedding, k=1)
    search_ms = (time.perf_counter() - search_start) * 1000

    distance = float(distances[0][0])
    idx = int(indices[0][0])
    confidence = max(0, 1 - distance)  # Convert distance to confidence

    total_ms = (time.perf_counter() - total_start) * 1000

    if confidence >= CONFIDENCE_THRESHOLD and idx in person_lookup:
        person = person_lookup[idx]
        attendance = record_attendance(
            person["id"], request.location_id, request.timestamp
        )
        return DetectResponse(
            success=True,
            person=person,
            confidence=round(confidence, 3),
            embedding_time_ms=round(embed_ms, 1),
            search_time_ms=round(search_ms, 1),
            total_time_ms=round(total_ms, 1),
            attendance=attendance,
        )
    else:
        return DetectResponse(
            success=False,
            person=None,
            confidence=round(confidence, 3),
            embedding_time_ms=round(embed_ms, 1),
            search_time_ms=round(search_ms, 1),
            total_time_ms=round(total_ms, 1),
            attendance={"status": "unknown", "is_late": False},
        )


@app.get("/api/v1/health")
async def health():
    return {
        "status": "ok",
        "gpu_available": True,
        "faiss_index_size": index.ntotal,
        "uptime_seconds": get_uptime(),
        "queue_depth": 0,
        "avg_latency_ms": get_avg_latency(),
    }
```

---

## Quick Start Checklist

- [ ] Install React Native + VisionCamera dependencies
- [ ] Implement native face detection frame processor (Android + iOS)
- [ ] Configure API base URL (Jetson local IP or cloud VM URL)
- [ ] Set up certificate pinning for production
- [ ] Test face detection accuracy with different lighting conditions
- [ ] Test auto-capture flow end-to-end
- [ ] Configure offline queue for unreliable networks
- [ ] Add liveness detection (blink + head movement at minimum)
- [ ] Comply with biometric data regulations (BIPA / GDPR Art. 9)
- [ ] Load test with 200 concurrent devices against Jetson
- [ ] Set up Grafana dashboard for monitoring latency + error rates
- [ ] Add enrollment screen for new face registration (3-5 angle captures)

---

## Appendix: API Endpoint Summary

| Method | Endpoint | Description | Auth |
|--------|----------|-------------|------|
| `POST` | `/api/v1/detect` | Recognize face, record attendance | Bearer token |
| `POST` | `/api/v1/register` | Enroll new face (3-5 images) | Bearer token + admin |
| `GET` | `/api/v1/health` | Server health + GPU status | None |
| `GET` | `/api/v1/attendance` | Today's attendance records | Bearer token |
| `DELETE` | `/api/v1/faces/{id}` | Remove face from index | Bearer token + admin |
| `PUT` | `/api/v1/faces/{id}` | Update face embeddings | Bearer token + admin |

---

*Built for the Mobile-to-Jetson Face Recognition Architecture.*
*Compatible with Jetson Nano (edge) and GPU VM (cloud) deployments.*
