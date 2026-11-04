''' hay q programar algo de temperaturas que no sale en el codigo original
    algo de colores y temperaturas '''

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

class Horno:
    def __init__(self, geom_id, temperatura_inicial=20.0):
        self.numero_piezas = 0
        self.temperatura = temperatura_inicial
        self.temperatura_min = 20.0
        self.temperatura_max = 150.0
        self.geom_id = geom_id
        self.mat_id = None  # ID del material incandescente

    def registrar_pieza(self):
        self.numero_piezas += 1

    def aumentar_temperatura(self, incremento):
        self.temperatura = min(self.temperatura_max, self.temperatura + incremento)

    def disminuir_temperatura(self, decremento):
        self.temperatura = max(self.temperatura_min, self.temperatura - decremento)

    def obtener_temperatura(self):
        return self.temperatura

    def actualizar_color(self, model):
        if self.temperatura < 50.0:
            color = [0.1, 0.3, 1.0, 1.0] # Azul (frío)
        elif self.temperatura < 100.0:
            color = [1.0, 0.6, 0.1, 1.0] # Naranja (templado)
        else:
            color = [1.0, 0.1, 0.1, 1.0] # Rojo vivo (caliente)

        # Modificamos el color del material directamente en el modelo de MuJoCo
        if model is not None and self.mat_id is not None and self.mat_id >= 0:
            model.mat_rgba[self.mat_id] = color