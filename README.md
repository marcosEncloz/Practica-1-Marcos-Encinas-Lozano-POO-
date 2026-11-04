# Practica-1-Marcos-Encinas-Lozano-POO-
# Planta Industrial - Robótica UBU

Simulación de planta industrial automatizada con MuJoCo, OpenCV y robot Franka Emika Panda (Programación Orientada a Objetos en Python).

## Estructura
- main.py: Lanzador principal de la simulación
- src/planta.py: Clase coordinadora y bucle principal
- src/cinta_transportadora.py: Control de cintas
- src/camara_inspeccion.py: Procesamiento con OpenCV
- src/horno.py: Gestión térmica y de piezas
- src/objeto_industrial.py: Representación de piezas
- src/robot.py: Cinemática inversa (IK) del robot

## Instalación de dependencias
pip install mujoco opencv-python numpy

## Ejecución
python main.py

## Controles por teclado
- 1 a 5: Activar / desactivar cintas transportadoras
- P: Pausar / reanudar simulación
- R: Reiniciar planta y contadores
