#include <DHT.h>
#define DHT_PIN 4
#define DHT_TYPE DHT11
#define SOIL_PIN 34
#define SOIL_DRY 3200
#define SOIL_WET 1400

DHT dht(DHT_PIN, DHT_TYPE);

void setup() {
  Serial.begin(9600);
  dht.begin();
  delay(1000);
}

void loop() {
  float humidity = dht.readHumidity();
  float temperature = dht.readTemperature();
  int raw = analogRead(SOIL_PIN);
  float moisture = 100.0 * (SOIL_DRY - raw) / (SOIL_DRY - SOIL_WET);
  moisture = constrain(moisture, 0, 100);
  if (isnan(humidity) || isnan(temperature)) {
    Serial.println("ERROR");
  } else {
    Serial.print(temperature, 1);
    Serial.print(",");
    Serial.print(moisture, 1);
    Serial.print(",");
    Serial.println(humidity, 1);
  }
  delay(3000);
}