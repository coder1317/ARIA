---
description: ESP32 and Arduino embedded development with sensors and IoT
triggers: esp32, arduino, embedded, iot, sensor, microcontroller
---

# ESP32/Arduino Embedded Skill

## Core Stack
- ESP32 DevKit V1 (WiFi + Bluetooth + 520KB SRAM)
- Arduino framework (PlatformIO or Arduino IDE)
- Libraries: WiFi, MQTT, HTTPClient, ArduinoJson, PubSubClient

## Template: ESP32 WiFi + MQTT
```cpp
#include <WiFi.h>
#include <PubSubClient.h>
#include <ArduinoJson.h>

const char* ssid = "YOUR_WIFI";
const char* password = "YOUR_PASS";
const char* mqtt_server = "broker.hivemq.com";

WiFiClient espClient;
PubSubClient client(espClient);

void setup() {
    Serial.begin(115200);
    WiFi.begin(ssid, password);
    while (WiFi.status() != WL_CONNECTED) delay(500);
    client.setServer(mqtt_server, 1883);
}

void loop() {
    if (!client.connected()) {
        // reconnect logic
    }
    client.loop();
    // publish sensor data every 5s
    static unsigned long lastPub = 0;
    if (millis() - lastPub > 5000) {
        float temp = readSensor();
        char payload[64];
        snprintf(payload, sizeof(payload), "{\"temp\":%.1f}", temp);
        client.publish("sensors/esp32", payload);
        lastPub = millis();
    }
}
```

## Common Sensors
- DHT22: Temperature + humidity (GPIO4)
- BMP280: Pressure + temperature (I2C)
- HC-SR04: Ultrasonic distance (GPIO trigger + echo)
- ADS1115: 16-bit ADC (I2C, for analog sensors)

## Wiring Rules
- Always use 3.3V for ESP32 (NOT 5V)
- Pull-up resistors on I2C (SDA/SCL)
- Decoupling capacitor (100nF) on power pins
- Use `INPUT_PULLUP` for buttons
