#!/usr/bin/env python3
# -*- coding: utf-8 -*-

class ObjetoIndustrial:

    def __init__(self, id_obj, tipo, body_name, segmento=0):
        self.id = id_obj
        self.tipo = tipo
        self.body_name = body_name
        self.segmento = segmento

    def actualizar_segmento(self, nuevo_segmento):
        self.segmento = nuevo_segmento

    def obtener_tipo(self):
        return self.tipo