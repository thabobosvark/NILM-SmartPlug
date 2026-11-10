#include <WiFi.h>
#include <PubSubClient.h>

// ---------------- WiFi ----------------
const char* ssid = "******";
const char* password = "******";

// ---------------- MQTT broker (your laptop's LAN IP) ----------------
const char* mqtt_server = "******";
const int mqtt_port = 1883;
const char* topic_publish = "nilm/plug1/reading";
const char* topic_subscribe = "nilm/plug1/result";

// ---------------- Hardware Serial 2 Pins (From Resistor Divider) ----------------
#define RXD2 16
#define TXD2 17

WiFiClient espClient;
PubSubClient client(espClient);

unsigned long lastPublish = 0;
const unsigned long PUBLISH_INTERVAL_MS = 7000;

unsigned long lastMqttRetry = 0;
const unsigned long MQTT_RETRY_INTERVAL_MS = 3000; // Try connecting every 3s WITHOUT freezing

// Holding variables for the extracted data stream
float v = 0.0, i = 0.0, w = 0.0, pf = 0.0, f = 0.0, e = 0.0;

// Called whenever a classification result string returns from your laptop's Python script
void callback(char* topic, byte* payload, unsigned int length) {
  String message;
  for (unsigned int i = 0; i < length; i++) message += (char)payload[i];

  Serial.print("AI Inference Result: ");
  Serial.println(message);
}

// FIX: Non-blocking MQTT reconnection strategy
void checkMQTTConnection() {
  if (client.connected()) return;

  unsigned long currentMillis = millis();
  if (currentMillis - lastMqttRetry >= MQTT_RETRY_INTERVAL_MS) {
    lastMqttRetry = currentMillis;
    
    Serial.print("Attempting non-blocking MQTT connection...");
    if (client.connect("ESP32SmartPlug")) {
      Serial.println("connected cleanly!");
      client.subscribe(topic_subscribe);
    } else {
      Serial.print("failed, rc=");
      Serial.println(client.state());
    }
  }
}

void setup() {
  Serial.begin(115200);
  
  // Listen for the incoming 5V converted telemetry packets on Serial2
  Serial2.begin(9600, SERIAL_8N1, RXD2, TXD2); 

  WiFi.begin(ssid, password);
  Serial.print("Connecting to WiFi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nConnected to WiFi");
  Serial.print("IP Address: ");
  Serial.println(WiFi.localIP());

  client.setServer(mqtt_server, mqtt_port);
  client.setCallback(callback);
}

void loop() {
  // Maintain network states seamlessly without hard delay loops
  checkMQTTConnection();
  if (client.connected()) {
    client.loop();
  }

  // --- Step A: Continuously capture and update data from the Arduino ---
  if (Serial2.available()) {
    String incomingData = Serial2.readStringUntil('\n');
    incomingData.trim();

    // Parse the comma-separated tokens
    int idx1 = incomingData.indexOf(',');
    int idx2 = incomingData.indexOf(',', idx1 + 1);
    int idx3 = incomingData.indexOf(',', idx2 + 1);
    int idx4 = incomingData.indexOf(',', idx3 + 1);
    int idx5 = incomingData.indexOf(',', idx4 + 1);

    if (idx1 > 0 && idx2 > idx1 && idx3 > idx2 && idx4 > idx3 && idx5 > idx4) {
      float tempV  = incomingData.substring(0, idx1).toFloat();
      float tempI  = incomingData.substring(idx1 + 1, idx2).toFloat();
      float tempW  = incomingData.substring(idx2 + 1, idx3).toFloat();
      float tempPF = incomingData.substring(idx3 + 1, idx4).toFloat();
      float tempF  = incomingData.substring(idx4 + 1, idx5).toFloat();
      float tempE  = incomingData.substring(idx5 + 1).toFloat();

      // GLITCH FILTER: Suppress serial noise spikes, accept standard voltage parameters
      if (tempV >= 50.0 && tempV <= 300.0 || 0.0) {
         v = tempV; i = tempI; w = tempW; pf = tempPF; f = tempF; e = tempE;
         
         // Visual confirmation to console that data parsed cleanly
         Serial.print("SUCCESS -> Voltage: ");
         Serial.print(v);
         Serial.print("V | Current: ");
         Serial.print(i);
         Serial.println("A");
      }
    }
  }

  // --- Step B: Controlled MQTT execution interval (Every 7 seconds) ---
  unsigned long now = millis();
  if (now - lastPublish >= PUBLISH_INTERVAL_MS) {
    lastPublish = now;

    // Only attempt publish if MQTT is actually connected
    if (client.connected()) {
      char payload[200];
      snprintf(payload, sizeof(payload),
        "{\"device_id\":\"plug1\",\"V\":%.1f,\"I\":%.3f,\"W\":%.1f,\"PF\":%.2f,\"Hz\":%.1f,\"kWh\":%.3f}",
        v, i, w, pf, f, e);

      Serial.print("Publishing telemetry payload: ");
      Serial.println(payload);
      client.publish(topic_publish, payload);
    }
  }
}
