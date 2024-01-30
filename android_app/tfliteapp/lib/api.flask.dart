import 'dart:convert';
import 'dart:io';

import 'package:camera/camera.dart';
import 'package:http/http.dart' as http;

var fileName = "";

//base64?
Future data(File image) async {
  var timestamp = DateTime.now().toString();
  var url = 'http://10.0.2.2:5000/upload';
  var uri = Uri.parse(url);
  var request = http.MultipartRequest("POST", uri);

  Map<String, String> headers = {"Content-type": "multipart/form-data"};

  fileName = image.path.split('/').last;
  //request.fields["inputPicture"] = "inputPicture" + timestamp;
  request.files.add(
    http.MultipartFile(
      'files[]',
      image.readAsBytes().asStream(),
      image.lengthSync(),
      filename: fileName,
    ),
  );
  request.headers.addAll(headers);
  var res = await request.send();

  http.Response response = await http.Response.fromStream(res);

  //http.Response response = await http.get(uri);
  var data = response.body;
  var decodedData = jsonDecode(data);
  return (decodedData['message'].toString());
  //return response.toString();
}

Future getData() async {
  var url = 'http://10.0.2.2:5000/get-prediction-flutter/' + fileName;
  var uri = Uri.parse(url);

  http.Response response = await http.get(uri);
  var data = response.body;
  var decodedData = jsonDecode(data);
  return (decodedData['flutterpred'].toString());
  //return response.toString();
}

Future getPredictionPDF() async {
  File file = new File("output.pdf");

  var url = 'http://10.0.2.2:5000/get-predictions';
  var uri = Uri.parse(url);

  try {
    await http.get(uri, headers: {"Content-Type": "application/json"}).then(
        (response) async {
      await file.writeAsBytes(response.bodyBytes);
    });
  } catch (Exception) {
    print(Exception.toString());
  }
  return file;
}
