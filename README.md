/*
 * ESP32-CAM Object Detection using Edge Impulse
 * by CircuitDigest (modified clean version)
 */

#include <your_project_name_inferencing.h>   // <-- CHANGE THIS to your model file
#include "edge-impulse-sdk/dsp/image/image.hpp"
#include "esp_camera.h"
#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h"

// ================= CAMERA MODEL =================
#define CAMERA_MODEL_AI_THINKER

#if defined(CAMERA_MODEL_AI_THINKER)
#define PWDN_GPIO_NUM     32
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM      0
#define SIOD_GPIO_NUM     26
#define SIOC_GPIO_NUM     27

#define Y9_GPIO_NUM       35
#define Y8_GPIO_NUM       34
#define Y7_GPIO_NUM       39
#define Y6_GPIO_NUM       36
#define Y5_GPIO_NUM       21
#define Y4_GPIO_NUM       19
#define Y3_GPIO_NUM       18
#define Y2_GPIO_NUM        5
#define VSYNC_GPIO_NUM    25
#define HREF_GPIO_NUM     23
#define PCLK_GPIO_NUM     22
#endif

// ================= OLED =================
#define I2C_SDA 15
#define I2C_SCL 14

TwoWire I2Cbus = TwoWire(0);

#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 64
#define OLED_RESET -1
#define SCREEN_ADDRESS 0x3C

Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &I2Cbus, OLED_RESET);

// ================= IMAGE BUFFER =================
#define EI_CAMERA_RAW_FRAME_BUFFER_COLS 320
#define EI_CAMERA_RAW_FRAME_BUFFER_ROWS 240
#define EI_CAMERA_FRAME_BYTE_SIZE 3

static bool is_initialised = false;
static bool debug_nn = false;

uint8_t *snapshot_buf;

// ================= CAMERA CONFIG =================
static camera_config_t camera_config = {
  .pin_pwdn = PWDN_GPIO_NUM,
  .pin_reset = RESET_GPIO_NUM,
  .pin_xclk = XCLK_GPIO_NUM,
  .pin_sscb_sda = SIOD_GPIO_NUM,
  .pin_sscb_scl = SIOC_GPIO_NUM,

  .pin_d7 = Y9_GPIO_NUM,
  .pin_d6 = Y8_GPIO_NUM,
  .pin_d5 = Y7_GPIO_NUM,
  .pin_d4 = Y6_GPIO_NUM,
  .pin_d3 = Y5_GPIO_NUM,
  .pin_d2 = Y4_GPIO_NUM,
  .pin_d1 = Y3_GPIO_NUM,
  .pin_d0 = Y2_GPIO_NUM,

  .pin_vsync = VSYNC_GPIO_NUM,
  .pin_href = HREF_GPIO_NUM,
  .pin_pclk = PCLK_GPIO_NUM,

  .xclk_freq_hz = 20000000,
  .ledc_timer = LEDC_TIMER_0,
  .ledc_channel = LEDC_CHANNEL_0,

  .pixel_format = PIXFORMAT_JPEG,
  .frame_size = FRAMESIZE_QVGA,
  .jpeg_quality = 12,
  .fb_count = 1,
  .fb_location = CAMERA_FB_IN_PSRAM,
  .grab_mode = CAMERA_GRAB_WHEN_EMPTY
};

// ================= FUNCTION DECLARATIONS =================
bool ei_camera_init(void);
bool ei_camera_capture(uint32_t img_width, uint32_t img_height, uint8_t *out_buf);
static int ei_camera_get_data(size_t offset, size_t length, float *out_ptr);

// ================= SETUP =================
void setup() {
  Serial.begin(115200);

  I2Cbus.begin(I2C_SDA, I2C_SCL, 100000);

  if (!display.begin(SSD1306_SWITCHCAPVCC, SCREEN_ADDRESS)) {
    Serial.println("OLED init failed!");
    while (1);
  }

  display.clearDisplay();
  display.setTextSize(1);
  display.setTextColor(WHITE);
  display.setCursor(0, 0);
  display.println("Starting...");
  display.display();

  if (!ei_camera_init()) {
    Serial.println("Camera init failed!");
  } else {
    Serial.println("Camera ready");
  }

  delay(2000);
}

// ================= LOOP =================
void loop() {

  snapshot_buf = (uint8_t *)malloc(EI_CAMERA_RAW_FRAME_BUFFER_COLS *
                                   EI_CAMERA_RAW_FRAME_BUFFER_ROWS *
                                   EI_CAMERA_FRAME_BYTE_SIZE);

  if (!snapshot_buf) {
    Serial.println("Memory allocation failed!");
    return;
  }

  if (!ei_camera_capture(EI_CLASSIFIER_INPUT_WIDTH,
                         EI_CLASSIFIER_INPUT_HEIGHT,
                         snapshot_buf)) {
    Serial.println("Capture failed");
    free(snapshot_buf);
    return;
  }

  ei::signal_t signal;
  signal.total_length = EI_CLASSIFIER_INPUT_WIDTH * EI_CLASSIFIER_INPUT_HEIGHT;
  signal.get_data = &ei_camera_get_data;

  ei_impulse_result_t result = {0};

  EI_IMPULSE_ERROR err = run_classifier(&signal, &result, debug_nn);

  if (err != EI_IMPULSE_OK) {
    Serial.println("Classifier error");
    free(snapshot_buf);
    return;
  }

  display.clearDisplay();

#if EI_CLASSIFIER_OBJECT_DETECTION == 1

  bool found = false;

  for (size_t i = 0; i < result.bounding_boxes_count; i++) {
    auto bb = result.bounding_boxes[i];

    if (bb.value == 0) continue;

    found = true;

    Serial.printf("%s (%.2f)\n", bb.label, bb.value);

    display.setCursor(0, i * 15);
    display.setTextSize(1);
    display.print(bb.label);
    display.print(" ");
    display.print((int)(bb.value * 100));
    display.print("%");
  }

  if (!found) {
    display.setCursor(0, 20);
    display.print("No objects found");
  }

#endif

  display.display();
  free(snapshot_buf);
  delay(500);
}

// ================= CAMERA INIT =================
bool ei_camera_init(void) {

  if (is_initialised) return true;

  esp_err_t err = esp_camera_init(&camera_config);

  if (err != ESP_OK) {
    Serial.printf("Camera init failed: 0x%x\n", err);
    return false;
  }

  is_initialised = true;
  return true;
}

// ================= CAPTURE =================
bool ei_camera_capture(uint32_t img_width, uint32_t img_height, uint8_t *out_buf) {

  if (!is_initialised) return false;

  camera_fb_t *fb = esp_camera_fb_get();

  if (!fb) return false;

  bool converted = fmt2rgb888(fb->buf, fb->len, PIXFORMAT_JPEG, snapshot_buf);

  esp_camera_fb_return(fb);

  if (!converted) return false;

  return true;
}

// ================= DATA CALLBACK =================
static int ei_camera_get_data(size_t offset, size_t length, float *out_ptr) {

  size_t pixel_ix = offset * 3;

  for (size_t i = 0; i < length; i++) {

    out_ptr[i] =
      (snapshot_buf[pixel_ix + 2] << 16) +
      (snapshot_buf[pixel_ix + 1] << 8) +
      snapshot_buf[pixel_ix];

    pixel_ix += 3;
  }

  return 0;
}
