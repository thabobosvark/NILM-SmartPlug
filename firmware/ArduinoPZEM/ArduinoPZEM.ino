#include <SoftwareSerial.h>
#include <PZEM004Tv30.h>

// PZEM connections on Arduino (Pins 2 and 3)
PZEM004Tv30 pzem(2, 3); 

// Serial link to the ESP32 (Pins 10 and 11)
SoftwareSerial linkToESP(10, 11); 

void setup() {
  linkToESP.begin(9600); // Math link matching ESP32 HardwareSerial2
}

void loop() {
  // Query all 6 parameters required for your NILM JSON profile
  float v  = pzem.voltage();
  float i  = pzem.current();
  float w  = pzem.power();
  float e  = pzem.energy();
  float f  = pzem.frequency();
  float pf = pzem.pf();

  // Safety filter if the AC side is completely unplugged
  if (isnan(v) || isnan(i) || isnan(w)) {
    // Transmit zero values/error state so the ESP32 stays updated
    linkToESP.println("0.0,0.0,0.0,0.0,0.0,0.0");
  } 
  else {
    // Pack all metrics sequentially into a clean CSV string
    linkToESP.print(v, 1);   linkToESP.print(",");
    linkToESP.print(i, 3);   linkToESP.print(",");
    linkToESP.print(w, 1);   linkToESP.print(",");
    linkToESP.print(pf, 2);  linkToESP.print(",");
    linkToESP.print(f, 1);   linkToESP.print(",");
    linkToESP.println(e, 3); // Finish with a newline separator
  }
  
  // Collect and stream data aggressively every 1.5 seconds.
  // The ESP32's non-blocking interval timer will throttle MQTT to 7 seconds.
  delay(1500); 
}
