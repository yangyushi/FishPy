#include <math.h>
/*
 * Signal Generator for Synchronised Cameras, the signals
 *     will be constantly generated until the arduino is reset
 *     
 * Written by Yushi Yang (yangyushi1992@icluod.com)
 * 
 * Hardware:
 *  High signals were generated from digital pin 8, 9, and 10
 *  Signals from brightness sensor are coming from Analog In Pin A0
 *  Signals from sound sensor are coming from Analog In Pin A1
 *  
 * How to use:
 *  Send "start"    to start a movie record with fps = [frame_rate]
 *  Send "delay 10" to start a movie record after 10 seconds
 *  Send "fps 10"   to set the fps to be 10.
 */


// configuration of pins
const int light_pin = A0;
const int sound_pin = A1;
const int led_pin = 11;
const int signal_pins[3] = {8, 9, 10};  // numbers of pins that connect to camera


String input;
long delay_time;
bool should_start = 0;

double frame_rate = 15;
unsigned long time_on;
unsigned long time_off;
const double on_percent = 50;
double period;


void update(){
    period = 1000 / frame_rate;  // Hz --> ms
    time_on = period / 100 * on_percent;  // unit: ms
    time_off = period - time_on;
}

void setup(){
  update();
  pinMode(led_pin, OUTPUT);
  for (int i=0; i < 3; i++){
    pinMode(signal_pins[i], OUTPUT);
  }
  Serial.begin(9600);
  Serial.setTimeout(100);
}

void read_sensors(){
  double bright_level = analogRead(light_pin);
  double sound = analogRead(sound_pin);
  Serial.print("Brightness: ");
  Serial.print(bright_level);
  Serial.print(", Noise level: ");
  Serial.print(sound);
  Serial.print('\n');
}

void generate_signal() {
  while (true) {
    delay(time_off);
    for (int i=0; i < 3; i=i+1){
      digitalWrite(signal_pins[i], HIGH);
    }
    delay(time_on);
    for (int i=0; i < 3; i=i+1){
      digitalWrite(signal_pins[i], LOW);
    }
  }
}


void loop(){
    if (millis() % 1000 < 100) {  // record every minute
        read_sensors();
    }
    input = Serial.readString();
    if (input == "start") {
        should_start = 1;
    } else if (input.startsWith("delay")){
        delay_time = input.substring(6).toInt();
        Serial.print("delaying ");
        Serial.print(delay_time);
        Serial.println(" s");
        delay_time = delay_time * 1000;
        delay(delay_time);
        should_start = 1;
    } else if (input.startsWith("fps")) {
        frame_rate = input.substring(4).toFloat();
        update();
        Serial.print("Setting fps to ");
        Serial.println(frame_rate);
    };
    if (should_start) {
        Serial.print("\nStart Recording (FPS=");
        Serial.print(frame_rate);
        Serial.println(")");
        generate_signal();
    }
}
