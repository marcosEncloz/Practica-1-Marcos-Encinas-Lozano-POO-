#!/usr/bin/env python3
# -*- coding: utf-8 -*-

class CintaTransportadora:

    def __init__(self, id_cinta, nombre, velocidad):
        self.id = id_cinta
        self.nombre = nombre
        self.velocidad = velocidad
        self.activa = True

    def arrancar(self):
        self.activa = True

    def detener(self):
        self.activa = False

    def conmutar(self):
        self.activa = not self.activa
        return self.activa

    def esta_activa(self):
        return self.activa