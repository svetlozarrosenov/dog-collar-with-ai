#include <RCSwitch.h>

class RFReceiver {
    private:
        unsigned long oldValue = 0;
        unsigned long currentValue = 0;
        // unsigned long startId = 12947240;
        // unsigned long stopId = 12947234;
        unsigned long startId = 14004776;
        unsigned long stopId = 14004770;

        // unsigned long startId = 1621800;
        // unsigned long stopId = 1621794;
        bool isActive = false;
        RCSwitch mySwitch;
        int rxPin;

    public:
        RFReceiver(int pin) {
            rxPin = pin;
        }
        
        bool isAlarmRunning() {
          return this->currentValue == this->startId;
        }
        
        bool alarmStateChanged() {
            if(mySwitch.available()) {
                unsigned long value = mySwitch.getReceivedValue();
                //Serial.println(value);
                if(value != this->startId && value != this->stopId) {
                  return false;
                }

                if(this->currentValue != value) {
                    this->oldValue = this->currentValue;
                    this->currentValue = value;
                    mySwitch.resetAvailable();
                    return true;
                }
            }
            return false;
        }

        void begin() {
            Serial.begin(115200);
            mySwitch.enableReceive(rxPin);
            Serial.println("433MHz Decoder Started");
        }
};