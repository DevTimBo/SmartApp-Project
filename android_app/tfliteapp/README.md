# TF-Lite Demo

A small project to show tf lite working in Flutter.

## Tf-Lite Version:

after some research I found out why my projekt isnt working. The [tflite](https://pub.dev/packages/tflite) plugin isn't working no more you need to use the [tflite_v2](https://pub.dev/packages/tflite_v2) to make it work. 

## compile the android App:
1. go in to the projekt folder  
2. ```bash 
    flutter build apk --split-per-abi
    ```
3. now you got three apks in build/app/outputs/flutter-apk
4. use the apk for your device architecture 