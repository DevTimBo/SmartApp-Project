import 'dart:async';
import 'dart:io';

import 'package:camera/camera.dart';
import 'package:daemmungsapp/api.flask.dart';
import 'package:document_file_save_plus/document_file_save_plus.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:path_provider/path_provider.dart';
import 'package:tflite_v2/tflite_v2.dart';
import 'package:flutter_pdfview/flutter_pdfview.dart';

//mostly from: https://docs.flutter.dev/cookbook/plugins/picture-using-camera
//Modified

String _classifications = "try again";
String? _image;

Future<void> main() async {
  // Ensure that plugin services are initialized so that `availableCameras()`
  // can be called before `runApp()`
  WidgetsFlutterBinding.ensureInitialized();

  // Obtain a list of the available cameras on the device.
  final cameras = await availableCameras();

  // Get a specific camera from the list of available cameras.
  final firstCamera = cameras.first;

  runApp(
    MaterialApp(
      theme: ThemeData.dark(),
      home: TakePictureScreen(
        // Pass the appropriate camera to the TakePictureScreen widget.
        camera: firstCamera,
      ),
    ),
  );
}

// A screen that allows users to take a picture using a given camera.
class TakePictureScreen extends StatefulWidget {
  const TakePictureScreen({
    super.key,
    required this.camera,
  });

  final CameraDescription camera;

  @override
  TakePictureScreenState createState() => TakePictureScreenState();
}

class TakePictureScreenState extends State<TakePictureScreen> {
  late CameraController _controller;
  late Future<void> _initializeControllerFuture;

  @override
  void initState() {
    super.initState();
    //load tfLiteModel
    _loadModel().then((_) {});
    // To display the current output from the Camera,
    // create a CameraController.
    _controller = CameraController(
      // Get a specific camera from the list of available cameras.
      widget.camera,
      // Define the resolution to use.
      ResolutionPreset.medium,
    );

    // Next, initialize the controller. This returns a Future.
    _initializeControllerFuture = _controller.initialize();
  }

  @override
  void dispose() {
    // Dispose of the controller when the widget is disposed.
    _controller.dispose();
    super.dispose();
  }

  _loadModel() async {
    await Tflite.loadModel(
        model: 'assets/faceId.tflite', labels: 'assets/labels.txt');
  }

  _predict(File image) async {
    // final output = await Tflite.runModelOnImage(
    //   path: image?.path ?? "",
    //   threshold: 0.8,
    //   numResults: 1,
    //   imageMean: 127.5,
    //   imageStd: 127.5,
    // );

    final isImageSend = await data(image);
    final prediction = await getPredictionPDF();

    if (isImageSend == "Files successfully uploaded") {
      setState(() {
        _classifications = "Bafoeg Prediction:";
        _image = prediction;
      });
    } else {
      setState(() {
        _classifications = isImageSend.toString();
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Take a picture')),
      // You must wait until the controller is initialized before displaying the
      // camera preview. Use a FutureBuilder to display a loading spinner until the
      // controller has finished initializing.
      body: FutureBuilder<void>(
        future: _initializeControllerFuture,
        builder: (context, snapshot) {
          if (snapshot.connectionState == ConnectionState.done) {
            // If the Future is complete, display the preview.
            return CameraPreview(_controller);
          } else {
            // Otherwise, display a loading indicator.
            return const Center(child: CircularProgressIndicator());
          }
        },
      ),
      floatingActionButton: FloatingActionButton(
        // Provide an onPressed callback.
        onPressed: () async {
          // Take the Picture in a try / catch block. If anything goes wrong,
          // catch the error.
          try {
            // Ensure that the camera is initialized.
            await _initializeControllerFuture;

            // Attempt to take a picture and get the file `image`
            // where it was saved.
            final image = await _controller.takePicture();
            final imageAsFile = File(image.path);
            await _predict(imageAsFile);
            if (!mounted) return;

            // If the picture was taken, display it on a new screen.
            await Navigator.of(context).push(
              MaterialPageRoute(
                builder: (context) => DisplayPictureScreen(
                  // Pass the automatically generated path to
                  // the DisplayPictureScreen widget.
                  imagePath: image.path,
                ),
              ),
            );
          } catch (e) {
            // If an error occurs, log the error to the console.
            print(e);
          }
        },
        child: const Icon(Icons.camera_alt),
      ),
    );
  }
}

// A widget that displays the picture taken by the user.
class DisplayPictureScreen extends StatelessWidget {
  final String imagePath;

  const DisplayPictureScreen({super.key, required this.imagePath});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text(_classifications)),
      // The image is stored as a file on the device. Use the `Image.file`
      // constructor with the given path to display the image.
      //body: Image.file(File(imagePath)),
      body: PDFView(
        filePath: _image,
      ),
      /* floatingActionButton: FloatingActionButton(
        // Provide an onPressed callback.
        onPressed: () async {
          // Take the Picture in a try / catch block. If anything goes wrong,
          // catch the error.
          try {
            final directory = await getExternalStorageDirectory();
            final file = File("${directory?.path}/example.pdf");

            final pdfBytes = await _image!.save();
            await _image!.;

            DocumentFileSavePlus().saveMultipleFiles(
              dataList: [
                pdfBytes,
              ],
              fileNameList: [
                "example.pdf",
              ],
              mimeTypeList: [
                "example/pdf",
              ],
            );
          } catch (e) {
            // If an error occurs, log the error to the console.
            print(e);
          }
        },
        child: const Icon(Icons.camera_alt),
      ), */
    );
  }
}
