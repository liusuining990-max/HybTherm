/*
  hybtherm_logger.ino — multi-probe DS18B20 logger for HybTherm

  Hardware (not required to use the rest of this repository):
    * Arduino UNO R3 or compatible (5 V)
    * up to 8 DS18B20 probes, all DATA lines on pin 2
    * one 4.7 kOhm pull-up resistor between DATA and 5 V
    * probes powered normally (red = 5 V, black = GND); parasite power is not used

  Output on the serial port at 115200 baud:
    # n_probes=5
    # probe1=28FF4A1B...   (64-bit ROM address of each probe, in bus order)
    time_s,T1,T2,T3,T4,T5
    0.0,22.1250,22.0625,...
    1.0,...

  One row per second. Conversions are started without blocking
  (setWaitForConversion(false)); a 12-bit conversion needs up to 750 ms, so the
  1 s period leaves margin. A failed read prints the library's error value
  -127.0, which the Python reader converts to NaN.

  The order of probes on the bus is fixed by their ROM addresses, not by where
  they sit on the block. Identify each probe once (warm one at a time with your
  hand and watch which column rises) and pass labels to the Python logger in
  that order.

  Status: written to match the OneWire / DallasTemperature library APIs; it has
  NOT yet been compiled or run on hardware by the maintainer.

  Libraries: OneWire (Paul Stoffregen), DallasTemperature (Miles Burton).
  License: MIT (see LICENSE in the repository root).
*/

#include <OneWire.h>
#include <DallasTemperature.h>

const uint8_t ONE_WIRE_PIN = 2;
const uint8_t MAX_PROBES = 8;
const unsigned long PERIOD_MS = 1000;

OneWire oneWire(ONE_WIRE_PIN);
DallasTemperature sensors(&oneWire);
DeviceAddress addr[MAX_PROBES];
uint8_t nProbes = 0;
unsigned long tStart = 0;
unsigned long tNext = 0;

void printAddress(const DeviceAddress a) {
  for (uint8_t i = 0; i < 8; i++) {
    if (a[i] < 16) Serial.print('0');
    Serial.print(a[i], HEX);
  }
}

void setup() {
  Serial.begin(115200);
  sensors.begin();
  uint8_t found = sensors.getDeviceCount();
  for (uint8_t i = 0; i < found && nProbes < MAX_PROBES; i++) {
    if (sensors.getAddress(addr[nProbes], i)) {
      sensors.setResolution(addr[nProbes], 12);
      nProbes++;
    }
  }
  sensors.setWaitForConversion(false);

  Serial.print(F("# n_probes="));
  Serial.println(nProbes);
  for (uint8_t i = 0; i < nProbes; i++) {
    Serial.print(F("# probe"));
    Serial.print(i + 1);
    Serial.print('=');
    printAddress(addr[i]);
    Serial.println();
  }
  Serial.print(F("time_s"));
  for (uint8_t i = 0; i < nProbes; i++) {
    Serial.print(F(",T"));
    Serial.print(i + 1);
  }
  Serial.println();

  sensors.requestTemperatures();  // first conversion
  tStart = millis();
  tNext = tStart + PERIOD_MS;
}

void loop() {
  unsigned long now = millis();
  if ((long)(now - tNext) < 0) return;
  tNext += PERIOD_MS;

  // Read the conversion started one period ago, then start the next one.
  Serial.print((now - tStart) / 1000.0, 1);
  for (uint8_t i = 0; i < nProbes; i++) {
    float c = sensors.getTempC(addr[i]);  // DEVICE_DISCONNECTED_C (-127) on error
    Serial.print(',');
    Serial.print(c, 4);
  }
  Serial.println();
  sensors.requestTemperatures();
}
