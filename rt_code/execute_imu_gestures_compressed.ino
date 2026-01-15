/***********************************************************/
/*  COMPRESSED MODEL VERSION (INT8 + 40% Pruned + EMA)    */
/***********************************************************/
/*
 * This sketch uses the compressed model with:
 * - 42 neurons in FC1 (40% pruned from 64)
 * - INT8 quantization
 * - EMA filter (alpha=0.3) applied BEFORE normalization
 *
 * Model file: model_weights_int8_pruned42.h
 *
 * PREPROCESSING PIPELINE (matches Python training):
 * 1. Read raw IMU data (6 channels: ax, ay, az, gx, gy, gz)
 * 2. Apply EMA filter with alpha=0.3 to all channels
 * 3. Normalize filtered values:
 *    - Accelerometer: divide by MAX_ACC (10 m/s²)
 *    - Gyroscope: divide by MAX_GYRO (2 rad/s)
 * 4. Feed to neural network for inference
 *
 * CRITICAL: EMA filter MUST be applied before normalization,
 * otherwise model performance will degrade (trained on filtered data).
 */

/***********************************************************/
/*                Libraries                                */
/***********************************************************/
#include <BLEDevice.h>
#include <BLEAdvertisedDevice.h>
#include <Adafruit_BNO08x.h>
#include "neural_network_engine.h"
// #include <ArduinoEigen.h>

/*************************************************************************************/
/*                                  Parameters                                   */
/*************************************************************************************/
// Bluetooth connection
#define ROBOT_ADDRESS  "24:62:ab:f2:af:46" // add here you devicec address
#define HAND_DIRECT_EXECUTE_SERVICE_UUID     "e0198000-7544-42c1-0000-b24344b6aa70"
#define EXECUTE_ON_WRITE_CHARACTERISTIC_UUID "e0198000-7544-42c1-0001-b24344b6aa70"
#define HAND_TRIGGER_SERVICE_UUID            "e0198002-7544-42c1-0000-b24344b6aa70"
#define TRIGGER_ON_WRITE_CHARACTERISTIC_UUID "e0198002-7544-42c1-0001-b24344b6aa70"
static BLEUUID serviceExeUUID(HAND_DIRECT_EXECUTE_SERVICE_UUID);// The remote service we wish to connect to.
static BLEUUID    charExeUUID(EXECUTE_ON_WRITE_CHARACTERISTIC_UUID);// The characteristic of the remote service we are interested in - execute.
static BLEUUID serviceTrigUUID(HAND_TRIGGER_SERVICE_UUID);// The remote service we wish to connect to.
static BLEUUID    charTrigUUID(TRIGGER_ON_WRITE_CHARACTERISTIC_UUID);// The characteristic of the remote service we are interested in - trigger

// Bluetooth connection status
static boolean doConnect = false;
static boolean isconnected = false;
// static boolean doScan = true;
static BLERemoteCharacteristic* pRemoteCharExecute;
static BLERemoteCharacteristic* pRemoteCharTrigger;
static BLEAdvertisedDevice* myDevice;
// scanning while disconnected
#define DT_SCAN    5000 //if not connected to BLE device scan every 5 seconds.
// #define DT_SLEEP    (10 * 60 * 1000) //if not connected to BLE device go to sleep after 10 minute.
unsigned long t_scan; //the elapsed time between BLE scanning
// unsigned long t_disconnected; //the elapsed time from a disconecting event

unsigned char* msg;
// uint8_t preset_id;

// the task the hand is required to fulfill
// movement length byte, torque, time, active motor, motor direction
unsigned char close_all_task[] = {5, 0b11111000, 20, 0b01111000, 0b11111000};
unsigned char open_all_task[] = {5, 0b11111000, 20, 0b01111000, 0b00000000};
unsigned char point_task[] = {5, 0b00000000, 20, 0b01111000, 0b10001000};
unsigned char turn_left_task[] = {5, 0b00000000, 25, 0b10000000, 0b10110000};
unsigned char turn_right_task[] = {5, 0b00000000, 25, 0b10000000, 0b00110000};

// gesture classifier neural engine
NeuralNetworkEngine nn_engine;
// Eigen::MatrixXf dummy_input;

// LED pins
#define GREEN_LED 2 // macro for green led color
#define BLUE_LED 4 // macro for blue led color
#define RED_LED 15 // macro for red led color
void setLedByClass(uint8_t class_index);
void setLedPins(uint8_t red_state, uint8_t green_state, uint8_t blue_state);

// IMU sensor properties
#define BNO08X_CS 10
#define BNO08X_INT 9
// For SPI mode, need a RESET, but not for I2C or UART
#define BNO08X_RESET -1
Adafruit_BNO08x bno08x(BNO08X_RESET);
sh2_SensorValue_t sensorValue;

// sensor buffer values - empty at first
#define BUFFER_SIZE 60
#define BUFFER_SIZE_TWICE BUFFER_SIZE * 2
#define NUM_CHANNELS 6
#define MAX_ACC 10 // normalization factor for accelerometer
#define MAX_GYRO 2 // normalization factor for gyro
Eigen::VectorXf buffer_vectors[NUM_CHANNELS];
Eigen::VectorXf filtered_buffer_vectors[NUM_CHANNELS];
Eigen::VectorXf imu_cache; // a cache to append IMU values
Eigen::MatrixXf input_tensor; // will be resized in setup

/*************************************************************************************/
/*                     EMA FILTER CONFIGURATION (NEW)                                */
/*************************************************************************************/
// EMA Filter Configuration (matches Python training: alpha=0.3)
const float EMA_ALPHA = 0.3;

// EMA filter state for 6 sensor channels (ax, ay, az, gx, gy, gz)
// Initialized to 0, first sample will initialize them
float ema_acc_x = 0.0;
float ema_acc_y = 0.0;
float ema_acc_z = 0.0;
float ema_gyro_x = 0.0;
float ema_gyro_y = 0.0;
float ema_gyro_z = 0.0;

// Flag to track first sample (for EMA initialization)
bool ema_initialized = false;

// regular operation
int counter;
uint8_t fingers_state = 0; // 0 - opened, 1 - closed, 2 - point (index finger only)
uint8_t wrist_state = 0; // 0 - straight, 1 - rotated
uint8_t previous_class_prediction = 0;

/*************************************************************************************/
/*                  BLE Class Definition and Set Callbacks:                          */
/*************************************************************************************/
class MyClientCallback : public BLEClientCallbacks {
  void onConnect(BLEClient* pclient) {
  }

  void onDisconnect(BLEClient* pclient) {

    isconnected = false;
    Serial.println("onDisconnect");
    // doScan = true;
    // t_disconnected = millis();
  }
};

bool connectToServer() {

    Serial.print("Forming a connection to ");
    Serial.println(myDevice->getAddress().toString().c_str());

    BLEClient*  pClient  = BLEDevice::createClient();
    Serial.println(" - Created client");

    pClient->setClientCallbacks(new MyClientCallback());

    // Connect to the remove BLE Server.
    pClient->connect(myDevice);  // if you pass BLEAdvertisedDevice instead of address, it will be recognized type of peer device address (public or private)
    Serial.println(" - Connected to server");

    // Execute Charachteristics:
    // Obtain a reference to the service we are after in the remote BLE server.
    BLERemoteService* pRemoteExeService = pClient->getService(serviceExeUUID);
    if (pRemoteExeService == nullptr) {
      Serial.print("Failed to find our Execute service UUID: ");
      Serial.println(serviceExeUUID.toString().c_str());
      pClient->disconnect();
      return false;
    }
    Serial.println(" - Found our Execute service");
    // Obtain a reference to the characteristic in the service of the remote BLE server.
    pRemoteCharExecute = pRemoteExeService->getCharacteristic(charExeUUID);
    if (pRemoteCharExecute == nullptr) {
      Serial.print("Failed to find our Execute characteristic UUID: ");
      Serial.println(charExeUUID.toString().c_str());
      pClient->disconnect();
      return false;
    }
    Serial.println(" - Found our Execute characteristic");

    // Read the value of the characteristic.
    if(pRemoteCharExecute->canRead()) {
      // std::string value = pRemoteCharExecute->readValue();
      String value = pRemoteCharExecute->readValue();
      Serial.print("Execute: The characteristic value was: ");
      Serial.println(value.c_str());
    }

    // Trigger Charachteristics:
    // Obtain a reference to the service we are after in the remote BLE server.
    BLERemoteService* pRemoteTrigService = pClient->getService(serviceTrigUUID);
    if (pRemoteTrigService == nullptr) {
      Serial.print("Failed to find our Trigger service UUID: ");
      Serial.println(serviceTrigUUID.toString().c_str());
      pClient->disconnect();
      return false;
    }
    Serial.println(" - Found our Trigger service");
    // Obtain a reference to the characteristic in the service of the remote BLE server.
    pRemoteCharTrigger = pRemoteTrigService->getCharacteristic(charTrigUUID);
    if (pRemoteCharTrigger == nullptr) {
      Serial.print("Failed to find our Trigger characteristic UUID: ");
      Serial.println(charTrigUUID.toString().c_str());
      pClient->disconnect();
      return false;
    }
    Serial.println(" - Found our Trigger characteristic");

    // Read the value of the characteristic.
    if(pRemoteCharTrigger->canRead()) {
      String value = pRemoteCharTrigger->readValue();
      Serial.print("Trigger: The characteristic value was: ");
      Serial.println(value.c_str());
    }

    isconnected = true;
    return isconnected;
}

// Scan for BLE servers and find the first one that advertises the service we are looking for.
class MyAdvertisedDeviceCallbacks: public BLEAdvertisedDeviceCallbacks {  // Called for each advertising BLE server.
  void onResult(BLEAdvertisedDevice advertisedDevice) {
    Serial.print("BLE Advertised Device found: ");
    Serial.println(advertisedDevice.toString().c_str());

    // We have found a device, let us now see if it contains the service we are looking for.
    if (((String)advertisedDevice.getAddress().toString().c_str()).equals(ROBOT_ADDRESS)) {

      BLEDevice::getScan()->stop();
      myDevice = new BLEAdvertisedDevice(advertisedDevice);
      doConnect = true;
      // doScan = false;

    } // Found our server
  } // onResult
}; // MyAdvertisedDeviceCallbacks

void InitBLE() {
  BLEDevice::init("SwitchLeg");
  // Retrieve a Scanner and set the callback we want to use to be informed when we
  // have detected a new device.  Specify that we want active scanning and start the
  // scan to run for 5 seconds.
  BLEScan* pBLEScan = BLEDevice::getScan();
  pBLEScan->setAdvertisedDeviceCallbacks(new MyAdvertisedDeviceCallbacks());
  pBLEScan->setInterval(1349);
  pBLEScan->setWindow(449);
  pBLEScan->setActiveScan(true);
  pBLEScan->start(1, false);
}

/*************************************************************************************/
/*                         EMA FILTER FUNCTIONS (NEW)                                */
/*************************************************************************************/
/**
 * Apply EMA (Exponential Moving Average) filter to a single sensor value.
 * This matches the preprocessing used during model training.
 *
 * Formula: filtered[i] = alpha * raw[i] + (1 - alpha) * filtered[i-1]
 *
 * @param raw_value: Current raw sensor reading
 * @param prev_filtered: Previous filtered value (state)
 * @param alpha: Smoothing factor (0.3 for our model)
 * @return: New filtered value
 */
float apply_ema_filter(float raw_value, float prev_filtered, float alpha) {
    return alpha * raw_value + (1.0 - alpha) * prev_filtered;
}

/**
 * Update all 6 EMA filter states with new IMU readings.
 * Call this immediately after reading from IMU, before normalization.
 *
 * @param raw_acc_x/y/z: Raw accelerometer readings (m/s²)
 * @param raw_gyro_x/y/z: Raw gyroscope readings (rad/s)
 */
void update_ema_filters(float raw_acc_x, float raw_acc_y, float raw_acc_z,
                        float raw_gyro_x, float raw_gyro_y, float raw_gyro_z) {
    // First sample: initialize EMA state with raw values
    if (!ema_initialized) {
        ema_acc_x = raw_acc_x;
        ema_acc_y = raw_acc_y;
        ema_acc_z = raw_acc_z;
        ema_gyro_x = raw_gyro_x;
        ema_gyro_y = raw_gyro_y;
        ema_gyro_z = raw_gyro_z;
        ema_initialized = true;
        return;
    }

    // Apply EMA filter to all 6 channels
    ema_acc_x = apply_ema_filter(raw_acc_x, ema_acc_x, EMA_ALPHA);
    ema_acc_y = apply_ema_filter(raw_acc_y, ema_acc_y, EMA_ALPHA);
    ema_acc_z = apply_ema_filter(raw_acc_z, ema_acc_z, EMA_ALPHA);
    ema_gyro_x = apply_ema_filter(raw_gyro_x, ema_gyro_x, EMA_ALPHA);
    ema_gyro_y = apply_ema_filter(raw_gyro_y, ema_gyro_y, EMA_ALPHA);
    ema_gyro_z = apply_ema_filter(raw_gyro_z, ema_gyro_z, EMA_ALPHA);
}

/**
 * Reset EMA filter state (call after gesture detection to prevent state bleed)
 */
void reset_ema_filters() {
    ema_initialized = false;
    // State will be reinitialized on next sample
}

/************************************************************************/
/*                           IMU Functions:                             */
/************************************************************************/
void setReports(void) {
  Serial.println("Setting desired reports");
  if (!bno08x.enableReport(SH2_ACCELEROMETER)) {
    Serial.println("Could not enable accelerometer");
  }
  if (!bno08x.enableReport(SH2_GYROSCOPE_CALIBRATED)) {
    Serial.println("Could not enable gyroscope");
  }
}

/************************************************************************/
/*                     Setup and Loop Functions:                        */
/************************************************************************/
void setup() {
  Serial.begin(115200);
  Serial.println("Device setup started...");
  Serial.println("COMPRESSED MODEL VERSION: INT8 + 40% Pruned + EMA Filter");

  // prepare led pins
  pinMode(GREEN_LED, OUTPUT); // green
  pinMode(BLUE_LED, OUTPUT); // blue
  pinMode(RED_LED, OUTPUT); // red

  Serial.println("Adafruit BNO08x test!");

  // initialize IMU sensor
  Serial.println("Looking for BNO08x sensor...");
  while (!bno08x.begin_I2C(0x4B)) {
    Serial.println("Failed to find BNO08x chip");
    ESP.restart(); // restart device - IMU accessible after
    delay(5);
  }
  Serial.println("BNO08x Found!");
  setReports();

  // print network for inspection and initialize input
  // nn_engine.printNeuralNetwork();
  imu_cache.resize(NUM_CHANNELS);
  imu_cache.setZero();
  for (int i = 0; i < NUM_CHANNELS; i++) {
    filtered_buffer_vectors[i].resize(BUFFER_SIZE);
  }
  input_tensor.resize(NUM_CHANNELS, BUFFER_SIZE);

  // create BLE Device
  InitBLE();
  t_scan = millis();
  Serial.println("BLE initiated.");

  // enable deep sleep mode for the esp32:
  esp_sleep_enable_ext0_wakeup(GPIO_NUM_4, 1); //1 = High, 0 = Low , same GPIO as the button pin
  // t_disconnected = millis();

  // initate message
  msg = close_all_task;

  // for debug - prepare dummy vector for inference
  // float dummy_array[6][60];
  // for (int i = 0; i < 6; i++) {
  //   for (int j = 0; j < 60; j++) {
  //     if (i < 3) {
  //       dummy_array[i][j] = 0.5f;
  //     } else {
  //       dummy_array[i][j] = 0.5f;
  //     }
  //   }
  // }
  // dummy_input = Eigen::MatrixXf(6, 60);
  // for (int i = 0; i < 6; i++) {
  //   for (int j = 0; j < 60; j++) {
  //     dummy_input(i, j) = dummy_array[i][j];
  //   }
  // }

  // turn on led to notify everything is working and turn it back off
  setLedPins(LOW, HIGH, LOW);
  delay(500);
  setLedPins(LOW, LOW, LOW);
}

void loop() {

  // check if controller had to reset the IMU
  if (bno08x.wasReset()) {
    Serial.print("sensor was reset ");
    setReports();
  }

  if (!bno08x.getSensorEvent(&sensorValue)) {
    return;
  }

  // measure time of execution
  unsigned long startTime = micros(); // - start time

  int current_buffer_size = 0;
  switch (sensorValue.sensorId) {

  case SH2_ACCELEROMETER:
    // print values for inspection - remove in real-time
    // Serial.print("Accelerometer - x: ");
    // Serial.print(sensorValue.un.accelerometer.x);
    // Serial.print(" y: ");
    // Serial.print(sensorValue.un.accelerometer.y);
    // Serial.print(" z: ");
    // Serial.println(sensorValue.un.accelerometer.z);

    current_buffer_size = buffer_vectors[0].size();
    if (current_buffer_size < BUFFER_SIZE_TWICE) {

      // === MODIFIED: Apply EMA filter BEFORE normalization ===
      // 1. Get raw accelerometer values
      float raw_acc_x = sensorValue.un.accelerometer.x;
      float raw_acc_y = sensorValue.un.accelerometer.y;
      float raw_acc_z = sensorValue.un.accelerometer.z;

      // 2. Apply EMA filter (CRITICAL: matches Python training)
      update_ema_filters(raw_acc_x, raw_acc_y, raw_acc_z,
                        ema_gyro_x, ema_gyro_y, ema_gyro_z);

      // 3. Normalize FILTERED values (not raw!)
      imu_cache(0) = ema_acc_x / MAX_ACC;
      imu_cache(1) = ema_acc_y / MAX_ACC;
      imu_cache(2) = ema_acc_z / MAX_ACC;

      // resize all six vectors to add one element each and add cache values
      for (int i = 0; i < NUM_CHANNELS; i++) {
        buffer_vectors[i].conservativeResize(current_buffer_size + 1);
        buffer_vectors[i](current_buffer_size) = imu_cache(i);
      }
    }

    break;
  case SH2_GYROSCOPE_CALIBRATED:
    // print values for inspection - remove in real-time
    // Serial.print("Gyro - x: ");
    // Serial.print(sensorValue.un.gyroscope.x);
    // Serial.print(" y: ");
    // Serial.print(sensorValue.un.gyroscope.y);
    // Serial.print(" z: ");
    // Serial.println(sensorValue.un.gyroscope.z);

    current_buffer_size = buffer_vectors[3].size();
    if (current_buffer_size < BUFFER_SIZE_TWICE) {

      // === MODIFIED: Apply EMA filter BEFORE normalization ===
      // 1. Get raw gyroscope values
      float raw_gyro_x = sensorValue.un.gyroscope.x;
      float raw_gyro_y = sensorValue.un.gyroscope.y;
      float raw_gyro_z = sensorValue.un.gyroscope.z;

      // 2. Apply EMA filter
      update_ema_filters(ema_acc_x, ema_acc_y, ema_acc_z,
                        raw_gyro_x, raw_gyro_y, raw_gyro_z);

      // 3. Normalize FILTERED values
      imu_cache(3) = ema_gyro_x / MAX_GYRO;
      imu_cache(4) = ema_gyro_y / MAX_GYRO;
      imu_cache(5) = ema_gyro_z / MAX_GYRO;

      // resize all six vectors to add one element each and add cache values
      for (int i = 0; i < NUM_CHANNELS; i++) {
        buffer_vectors[i].conservativeResize(current_buffer_size + 1);
        buffer_vectors[i](current_buffer_size) = imu_cache(i);
      }
    }

    break;
  }

  if (doConnect == true) { //TRUE when we scanned and found the desired BLE server
    connectToServer();

    if (isconnected)
      Serial.println("We are now connected to the BLE Server."); // connect to the server. TRUE if connection was established
    else
      Serial.println("We have failed to connect to the server; there is nothin more we will do.");
    doConnect = false; //no need to reconnect to the server
  }

  // prepare input vector for inference
  if (buffer_vectors[0].size() == BUFFER_SIZE_TWICE) {

    // efficiently extract even-indexed elements
    for (int i = 0; i < NUM_CHANNELS; i++) {
      for (int j = 0; j < BUFFER_SIZE; j++) {
        filtered_buffer_vectors[i](j) = buffer_vectors[i](2 * j); // Access even indices directly
      }
    }

    // concatenate the vectors into the input tensor
    for (int i = 0; i < NUM_CHANNELS; i++) {
      input_tensor.row(i) = filtered_buffer_vectors[i].transpose(); // Transpose to make it a row
    }
    // perform inference.
    uint8_t class_index = nn_engine.predict(input_tensor);
    // int class_index = nn_engine.predict(dummy_input);

    // new gesture detected (avoid repeating predictions)
    if (class_index != previous_class_prediction) {
      // set LED to new status
      setLedByClass(class_index);

      // send intructions to the hand if connected
      if (class_index > 0 && isconnected) {

        // set the message to be sent according to the hand state
        // change fingers state
        bool new_state_requested = false;
        if (fingers_state != 1 && class_index == 1) { // close all fingers
          msg = close_all_task;
          fingers_state = 1;
          new_state_requested = true;
        } else if (fingers_state != 2 && class_index == 2) { // point with one finger
          msg = point_task;
          fingers_state = 2;
          new_state_requested = true;
        } else if ((fingers_state == 1 && class_index == 1) ||
                   (fingers_state == 2 && class_index == 2)) { // open all fingers
          msg = open_all_task;
          fingers_state = 0;
          new_state_requested = true;
        } else if (wrist_state == 0 && class_index == 3) { // rotate wrist to the write
          msg = turn_left_task;
          wrist_state = 1;
          new_state_requested = true;
        } else if (wrist_state == 1 && class_index == 4) { // rotate wrist to the left
          msg = turn_right_task;
          wrist_state = 0;
          new_state_requested = true;
        }

        if (new_state_requested) {
          pRemoteCharExecute->writeValue(msg,msg[0]);
          Serial.println("Message sent");
          // digitalWrite(LED_BUILTIN, HIGH);
          delay(1000);
        }
      } else { //not connected
        //scanning for server:
        if((millis()-t_scan>DT_SCAN)){ //not connected
          //BLEDevice::getScan()->start(0);  // start to scan after disconnect. 0 - scan for infinity, 1-1 sec, ..
          Serial.println("Scanning...");
          BLEDevice::getScan()->start(1, false);
          t_scan = millis();
        }
      }

      previous_class_prediction = class_index;
    }

    unsigned long endTime = micros();   // - end time
    unsigned long elapsedTime = endTime - startTime; // calculate elapsed time

    Serial.print("Elapsed Time: ");
    Serial.print(elapsedTime);
    Serial.println(" microseconds");

    // === MODIFIED: Reset EMA filters after gesture ===
    reset_ema_filters();

    // return back to data collection
    for (int i = 0; i < NUM_CHANNELS; i++) {
      buffer_vectors[i].resize(0); // empty the input vector
    }
    imu_cache.setZero(); // clear cache vector
  } else {
    delay(5);
  }

  // delay(100); // Optional delay to control how often sampling is performed
}

// set led color given class index
void setLedByClass(uint8_t class_index) {
  if (class_index == 1) {
    setLedPins(LOW, HIGH, LOW); // green
  } else if (class_index == 2) {
    setLedPins(LOW, HIGH, HIGH); // turquoise
  } else if (class_index == 3) {
    setLedPins(LOW, LOW, HIGH); // blue
  } else if (class_index == 4) {
    setLedPins(HIGH, LOW, HIGH); // purple
  } else { // class_index == 0
    setLedPins(LOW, LOW, LOW); // turned off
  }
}

void setLedPins(uint8_t red_state, uint8_t green_state, uint8_t blue_state) {
  digitalWrite(RED_LED, red_state); // set the value of the error led
  digitalWrite(GREEN_LED, green_state); // set the value of the green led
  digitalWrite(BLUE_LED, blue_state); // set the value of the blue led
}
