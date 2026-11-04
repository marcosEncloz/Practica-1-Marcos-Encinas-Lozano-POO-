#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np


class CamaraInspeccion:

    def __init__(self, nombre_mujoco, titulo_ventana, cinta_asociada):
        self.nombre = nombre_mujoco
        self.titulo = titulo_ventana
        self.cinta = cinta_asociada
        self.objeto_detectado = False
        self.forma_detectada = None
        self.ultima_imagen = None

    def _add_label(self, image, text, status_text=None):

        img = image.copy()

        cv2.rectangle(
            img,
            (0, 0),
            (img.shape[1], 36),
            (20, 20, 20),
            -1
        )

        cv2.putText(
            img,
            text,
            (12, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (245, 245, 245),
            2,
            cv2.LINE_AA
        )

        if status_text:

            cv2.putText(
                img,
                status_text,
                (12, img.shape[0] - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (245, 245, 245),
                1,
                cv2.LINE_AA
            )

        return img

    def capturar(self, renderer, data, status_text=None):
        renderer.update_scene(data, camera=self.nombre)
        rgb = renderer.render()
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        self.ultima_imagen = self._add_label(
            bgr,
            self.titulo,
            status_text
        )

        return self.ultima_imagen

    def detectar_forma(self, imagen):
        gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(
            gris,
            50,
            255,
            cv2.THRESH_BINARY
        )

        contornos, _ = cv2.findContours(
            thresh,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        for cnt in contornos:
            area = cv2.contourArea(cnt)
            if area > 500:
                perimetro = cv2.arcLength(cnt, True)
                aprox = cv2.approxPolyDP(
                    cnt,
                    0.04 * perimetro,
                    True
                )

                vertices = len(aprox)
                if vertices >= 8:
                    return "cilindro"
                elif vertices == 4:
                    return "rectangulo"
                else:
                    return "objeto"
        return None

    def actualizar(self, renderer, data, status_text=None):
        imagen = self.capturar(
            renderer,
            data,
            status_text
        )

        self.forma_detectada = self.detectar_forma(imagen)
        self.objeto_detectado = (
            self.forma_detectada is not None )

        if self.objeto_detectado and self.cinta:
            self.cinta.detener()
        return imagen

'''
class CamaraInspeccion:

    def __init__(self, nombre, cinta):
        self.nombre = nombre
        self.cinta = cinta
        self.objeto_detectado = False
        self.forma_detectada = None
    
    def capturar(self, renderer, data):
        ...
    
    def detectar_forma(self, imagen):
        ...
    
    def actualizar(self, renderer, data):
        imagen = self.capturar(renderer, data)
        
        self.forma_detectada = self.detectar_forma(imagen)
        self.objeto_detectado = self.forma_detectada is not None
    
        if self.objeto_detectado:
            self.cinta.detener()
    
        return imagen
    '''