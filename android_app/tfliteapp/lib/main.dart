// author Emil Hillebrand
import 'dart:async';
import 'dart:io';
import 'package:camera/camera.dart';
import 'package:daemmungsapp/api.flask.dart';
import 'package:flutter/material.dart';
// import 'package:tflite_v2/tflite_v2.dart';
import 'package:flutter_pdfview/flutter_pdfview.dart';

//inspired by: https://docs.flutter.dev/cookbook/plugins/picture-using-camera
//Modified

String _classifications = "try again";
String? _image;

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  // get available Cameras
  final cameras = await availableCameras();

  // Get a first Camera (experience. back Camera)
  final firstCamera = cameras.first;

  runApp(
    MaterialApp(
      theme: ThemeData.dark(),
      home: TakePictureScreen(
        camera: firstCamera,
      ),
    ),
  );
}

// screen to preview and take pictures
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
    //load tfLiteModel (not used old functions for TFLite functionality)
    //_loadModel().then((_) {});
    // To display the current output from the Camera,
    // create the CameraController.
    _controller = CameraController(
      // Get the choosen camera
      widget.camera,
      // resolution of Camera
      ResolutionPreset.medium,
    );

    // initialize controller
    _initializeControllerFuture = _controller.initialize();
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  // old function used for TFLite
  //_loadModel() async {
  // for the tfLite prototype
  //await Tflite.loadModel(
  //    model: 'assets/faceId.tflite', labels: 'assets/labels.txt');
  //}

  _predict(File image) async {
    // for the tfLite model now used for the API call
    // final output = await Tflite.runModelOnImage(
    //   path: image?.path ?? "",
    //   threshold: 0.8,
    //   numResults: 1,
    //   imageMean: 127.5,
    //   imageStd: 127.5,
    // );

    //Call the functions for the API call
    final isImageSend = await data(image);
    final prediction = await getPredictionPDF();

    //checking if everything worked smoothly
    if (isImageSend == "Files successfully uploaded") {
      setState(() {
        _classifications = "Bafoeg Prediction:";
        _image = prediction;
      });
      // sets the error directly to the title of the Shown ImageScreen
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
      body: FutureBuilder<void>(
        future: _initializeControllerFuture,
        builder: (context, snapshot) {
          if (snapshot.connectionState == ConnectionState.done) {
            // show preview
            return CameraPreview(_controller);
          } else {
            // show loading
            return const Center(child: CircularProgressIndicator());
          }
        },
      ),
      // button for taking picture
      floatingActionButton: FloatingActionButton(
        onPressed: () async {
          // taking the picture
          try {
            await _initializeControllerFuture;

            final image = await _controller.takePicture();
            final imageAsFile = File(image.path);
            // send and get the prediction from the API
            await _predict(imageAsFile);
            if (!mounted) return;
            // Displaying the PDF on new screen
            await Navigator.of(context).push(
              MaterialPageRoute(
                builder: (context) => DisplayPDFScreen(
                  imagePath: image.path,
                ),
              ),
            );
          } catch (e) {
            // error gets logged
            print(e);
          }
        },
        child: const Icon(Icons.camera_alt),
      ),
    );
  }
}

// A widget that displays the PDF taken by the user.
class DisplayPDFScreen extends StatelessWidget {
  final String imagePath;

  const DisplayPDFScreen({super.key, required this.imagePath});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text(_classifications)),
      // the PDF is stored on the Device. We now use the PDFView to display the PDF
      body: PDFView(
        filePath: _image,
      ),
    );
  }
}
