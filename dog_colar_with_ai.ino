// #include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
// #include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"  // ← за resolver (ако AllOps не работи, замени с това)

#include <Wire.h>

#include <Adafruit_ADXL345_U.h>

#include "model_data.h"   // твоят .tflite → .h/.cc

int int1Pin = 36;
int SDAPin = 34;
int SCLPin = 33;
int CSPin = 37;
int SDOPin = 35;

Adafruit_ADXL345_Unified accel = Adafruit_ADXL345_Unified(12345);
volatile bool motionDetected = false;

void IRAM_ATTR handleInterrupt() {
    motionDetected = true;
}

namespace {
  const tflite::Model* model = nullptr;
  tflite::MicroInterpreter* interpreter = nullptr;
  TfLiteTensor* input = nullptr;
  TfLiteTensor* output = nullptr;
  int inference_count = 0;

  constexpr int kTensorArenaSize = 85 * 1024;  // 85 KB – безопасно за твоя модел
  uint8_t tensor_arena[kTensorArenaSize];
}  // namespace

void setup() {
  Serial.begin(115200);
  while (!Serial);

  Serial.println("Зареждане на модел за тичане...");

  Wire.begin(SDAPin, SCLPin);

   if (!accel.begin()) {
    Serial.println("ADXL345 не е открит!");
  }

  setupADXL();

  // ← Твоята база – работи!
  model = tflite::GetModel(model_float32_tflite);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    MicroPrintf("Грешка: Модел версия %d (очаквана %d)", model->version(), TFLITE_SCHEMA_VERSION);
    while (1);
  }

  // ← НОВО: Mutable resolver (по-лек от AllOps; добави операции според модела ти)
 static tflite::MicroMutableOpResolver<4> resolver;
  resolver.AddFullyConnected();   // за всички Dense слоеве
  resolver.AddRelu();             // за relu активации
  resolver.AddAdd();              // за bias add
  resolver.AddLogistic();         // ← това е Sigmoid! (не Softmax!)

  // ← НОВО: Интерпретатор (без error_reporter – nullptr по подразбиране)
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize);  // ← точно този constructor работи!
  interpreter = &static_interpreter;

  // ← НОВО: Алоциране
  if (interpreter->AllocateTensors() != kTfLiteOk) {
    MicroPrintf("AllocateTensors failed! Arena: %d KB", kTensorArenaSize / 1024);
    while (1);
  }

  input = interpreter->input(0);
  output = interpreter->output(0);

  Serial.printf("Модел зареден! Вход: [%ldx%ldx%ld], Изход: %ld\n",
                input->dims->data[0], input->dims->data[1], input->dims->data[2],
                output->dims->size);
  Serial.println("Готов за детекция!");
}

void setupADXL() {
     // Обхват на ±4g (запазваме чувствителността)
    accel.setRange(ADXL345_RANGE_4_G);

    // Честота на измерване 100 Hz (запазваме)
    accel.writeRegister(ADXL345_REG_BW_RATE, 0x07); // 0x0A = 100 Hz

    // Праг за активност 2 единици (15.6 mg при ±4g, запазваме)
    accel.writeRegister(ADXL345_REG_THRESH_ACT, 2);

    // Активиране на всички оси
    accel.writeRegister(ADXL345_REG_ACT_INACT_CTL, 0xFF);
  
    // Прекъсване на INT1
    accel.writeRegister(ADXL345_REG_INT_MAP, 0x00);
    accel.writeRegister(ADXL345_REG_INT_ENABLE, 0x10);

    // Low Power режим за намалена консумация
    accel.writeRegister(ADXL345_REG_POWER_CTL, 0x08 | 0x10); // Measurement mode + Low Power

    // Изчистване на прекъсвания
    byte clearInt = accel.readRegister(ADXL345_REG_INT_SOURCE);
    
    Serial.print("ACT_INACT_CTL: 0x");
    Serial.println(accel.readRegister(ADXL345_REG_ACT_INACT_CTL), HEX);
    Serial.print("INT_ENABLE: 0x");
    Serial.println(accel.readRegister(ADXL345_REG_INT_ENABLE), HEX);
    Serial.print("INT_MAP: 0x");
    Serial.println(accel.readRegister(ADXL345_REG_INT_MAP), HEX);
    
    pinMode(int1Pin, INPUT);
    attachInterrupt(digitalPinToInterrupt(int1Pin), handleInterrupt, RISING);
}

void loop() {
  // Спи, докато няма движение
  if (!motionDetected) {
    delay(100);
    return;
  }

  motionDetected = false;  // изчистваме флага

  static float buffer[100][3];
  static int idx = 0;

  // Събираме точно 100 проби на ~100 Hz → 1 секунда прозорец
  for (int i = 0; i < 100; i++) {
    sensors_event_t event;
    accel.getEvent(&event);

    buffer[idx][0] = event.acceleration.x / 9.81f;  // в g
    buffer[idx][1] = event.acceleration.y / 9.81f;
    buffer[idx][2] = event.acceleration.z / 9.81f;

    idx = (idx + 1) % 100;

    // Копираме директно в input тензора (по-бързо)
    int pos = i * 3;
    input->data.f[pos + 0] = buffer[(idx + i) % 100][0];  // правилен ред
    input->data.f[pos + 1] = buffer[(idx + i) % 100][1];
    input->data.f[pos + 2] = buffer[(idx + i) % 100][2];

    delay(10);  // ~100 Hz
  }

  // Inference
  if (interpreter->Invoke() == kTfLiteOk) {
    float prob = output->data.f[0];
    inference_count++;
    Serial.printf("Инференс #%d → Тичане: %.1f%%\n", inference_count, prob * 100.0f);

    if (prob > 0.75f) {  // прага може да го настроиш по-късно
      Serial.println("ТИЧАНЕ!!!");
      // тук можеш да включиш buzzer, LoRa съобщение и т.н.
    }
  }

  // Изчистваме прекъсването
  accel.readRegister(ADXL345_REG_INT_SOURCE);
}