#include "LoRaWan_APP.h"
#include "Arduino.h"

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"  // ← за resolver (ако AllOps не работи, замени с това)

#include <Wire.h>

#include <Adafruit_ADXL345_U.h>

#include "model_data.h"   // твоят .tflite → .h/.cc

#define RF_FREQUENCY                                868000000 // Hz

#define TX_OUTPUT_POWER                             14        // dBm

#define LORA_BANDWIDTH                              0         // [0: 125 kHz,
                                                              //  1: 250 kHz,
                                                              //  2: 500 kHz,
                                                              //  3: Reserved]
#define LORA_SPREADING_FACTOR                       12         // [SF7..SF12]
#define LORA_CODINGRATE                             1         // [1: 4/5,
                                                              //  2: 4/6,
                                                              //  3: 4/7,
                                                              //  4: 4/8]
#define LORA_PREAMBLE_LENGTH                        8         // Same for Tx and Rx
#define LORA_SYMBOL_TIMEOUT                         0         // Symbols
#define LORA_FIX_LENGTH_PAYLOAD_ON                  false
#define LORA_IQ_INVERSION_ON                        false


#define RX_TIMEOUT_VALUE                            2000
#define BUFFER_SIZE                                 30 // Define the payload size here

char txpacket[BUFFER_SIZE];
char rxpacket[BUFFER_SIZE];

double txNumber;

bool lora_idle=true;

static RadioEvents_t RadioEvents;
void OnTxDone( void );
void OnTxTimeout( void );

static float buffer[200][3];  // 100 × 3 = точно колкото модела очаква
static int idx = 0;

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
  Mcu.begin(HELTEC_BOARD,SLOW_CLK_TPYE);

  btStop();

  Serial.println("Зареждане на модел за тичане...");


  txNumber = 0;

  RadioEvents.TxDone = OnTxDone;
  RadioEvents.TxTimeout = OnTxTimeout;

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

  static tflite::MicroMutableOpResolver<10> resolver;
    resolver.AddConv2D();
    resolver.AddDepthwiseConv2D();
    resolver.AddFullyConnected();
    resolver.AddRelu();
    resolver.AddLogistic();
    resolver.AddReshape();
    resolver.AddExpandDims();
    resolver.AddMean();
    resolver.AddMaxPool2D();   // ← задължително!
    resolver.AddPad();         // ← също често липсва

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

   txNumber=0;

    RadioEvents.TxDone = OnTxDone;
    RadioEvents.TxTimeout = OnTxTimeout;
    
    Radio.Init( &RadioEvents );
    Radio.SetChannel( RF_FREQUENCY );
    Radio.SetTxConfig( MODEM_LORA, TX_OUTPUT_POWER, 0, LORA_BANDWIDTH,
                                   LORA_SPREADING_FACTOR, LORA_CODINGRATE,
                                   LORA_PREAMBLE_LENGTH, LORA_FIX_LENGTH_PAYLOAD_ON,
                                   true, 0, 0, LORA_IQ_INVERSION_ON, 3000 ); 
}

void setupADXL() {
     // Обхват на ±4g (запазваме чувствителността)
    accel.setRange(ADXL345_RANGE_4_G);

    // Честота на измерване 100 Hz (запазваме)
    accel.writeRegister(ADXL345_REG_BW_RATE, 0x0A); // 0x0A = 100 Hz

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
  if (!motionDetected) { delay(50); return; }
  motionDetected = false;

  for (int i = 0; i < 200; i++) {
    sensors_event_t event;
    accel.getEvent(&event);

    buffer[idx][0] = event.acceleration.x / 1000.0f;
    buffer[idx][1] = event.acceleration.y / 1000.0f;
    buffer[idx][2] = event.acceleration.z / 1000.0f;

    idx = (idx + 1) % 200;        // ← 200 вместо 100
    delay(10);                    // точно 10 ms → 100 Hz → 200 точки = 2 секунди
}

  // Копиране в тензора
  for (int i = 0; i < 200; i++) {
    int src = (idx + i) % 200;               // ← 200
    input->data.f[i * 3 + 0] = buffer[src][0];
    input->data.f[i * 3 + 1] = buffer[src][1];
    input->data.f[i * 3 + 2] = buffer[src][2];
  }

  if (interpreter->Invoke() == kTfLiteOk) {
    lora_idle = true;
    float prob = output->data.f[0];
    Serial.printf("Тичане: %.2f%%\n", prob * 100.0f);
    String data = "";
    if (prob > 0.75) {
      Serial.println("RUNNING!!!");
      data = "RUNNING";
    } else {
      Serial.println("NOT RUNNING");
      data = "NOT RUNNING";
    }
    
    Serial.println("Sended data");
    Serial.println(data);
    Radio.Send((uint8_t *)data.c_str(), data.length());
  }

  accel.readRegister(ADXL345_REG_INT_SOURCE);
}

void OnTxDone(void) {
    Serial.println("TX done......");
    Radio.Sleep(); // Заспиване на LoRa след предаване
    lora_idle = true;
}

void OnTxTimeout(void) {
    Radio.Sleep(); // Заспиване при таймаут
    Serial.println("TX Timeout......");
    lora_idle = true;
}
