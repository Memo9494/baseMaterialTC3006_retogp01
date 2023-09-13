import network
import urequests
import machine
from machine import Pin, RTC, SoftI2C 
import ntptime
import time
from time import sleep

import dht                  # Librerias del sensor DTH11/22
from lcd_api import LcdApi  # Librerias de pantalla LCD
from i2c_lcd import I2cLcd

# Configuracion de la Red
wlan = network.WLAN(network.STA_IF)
wlan.active(True)
wlan.connect("Wokwi-GUEST","") # Setear con la Red Wifi (nombre, contraseña)
while not wlan.isconnected():
  pass

# Direccion de la base de datos en tiempo real
URL = "https://retoequipo1-3f172-default-rtdb.firebaseio.com/Reto_Equipo1/.json"
URL_hora = "https://retoequipo1-3f172-default-rtdb.firebaseio.com/Reto_Equipo1/hora.json"

# Inicializacion el sensor
sensor = dht.DHT22(Pin(4))

# Inicializacion de la pantalla LCD
I2C_ADDR = 0x27
totalRows = 4
totalColumns = 20

i2c = SoftI2C(scl=Pin(22), sda=Pin(21), freq=10000)     #initializing the I2C method for ESP32
#i2c = I2C(scl=Pin(5), sda=Pin(4), freq=10000)          #initializing the I2C method for ESP8266
lcd = I2cLcd(i2c, I2C_ADDR, totalRows, totalColumns)

# Funcion para imprimir en la LCD
def lcd_str(message, col, row):
    lcd.move_to(col, row) # Ubicacion
    lcd.putstr(message)   # Mensaje

while True:

    #Subir temperatura a la base de datos
    try:
        sleep(1)
        #Temperatura real - lectura sensor
        sensor.measure()
        temp = sensor.temperature()
        #Publicacion de la lectura
        temp_real_data = "{\"temp_real\":\"" + str(temp) + "\"}"
        temp_real = urequests.patch(URL, data=temp_real_data)
        res_real = temp_real.json()
        print(res_real)
        #Display
        lcd_str("Temperatura Real",0,0)
        lcd_str(str(temp),0,1)

    except OSError as e:
        print('Failed to read sensor.')

    #Request de estimacion
    try:
        #Request de la hora
        response = urequests.get(URL_hora)
        data = response.json()

        #Publicar Temperatura estimada por el modelo
        #temp_est_data = "{\"temp_est\":\"" + str(temp[0]) + "\"}"
        #temp_est = urequests.patch(URL, data=temp_est_data)
        #res_est = temp_est.json()
        #print(res_est)
        lcd_str("Temperatura Estimada:",0,2)
        lcd_str("29",0,3)

    except Exception as e:
        print(f"Error al obtener datos: {str(e)}")



