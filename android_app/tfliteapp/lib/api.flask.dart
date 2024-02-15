//author Emil Hillebrand
import 'dart:convert';
import 'dart:io';
import 'package:http/http.dart' as http;
import 'package:path_provider/path_provider.dart';

// in the earlier version we needed this because the get expected the same filename as the post now a relict to keep the old functoin functional if needed
var fileName = "";

//send the data to the API
Future data(File image) async {
  var url = 'http://10.0.2.2:5000/upload';
  var uri = Uri.parse(url);
  var request = http.MultipartRequest("POST", uri);

  Map<String, String> headers = {"Content-type": "multipart/form-data"};

  fileName = image.path.split('/').last;

  //we send the Picture as an multiPartFile (as a ByteStream)
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

  // we receive the massage and return the response.
  http.Response response = await http.Response.fromStream(res);
  var data = response.body;
  var decodedData = jsonDecode(data);
  return (decodedData['message'].toString());
}

//old function to get a textual prediction
Future getData() async {
  var url = 'http://10.0.2.2:5000/get-prediction-flutter/' + fileName;
  var uri = Uri.parse(url);

  http.Response response = await http.get(uri);
  var data = response.body;
  var decodedData = jsonDecode(data);
  return (decodedData['flutterpred'].toString());
}

//here we send an get to get the PDF the API is sending us
Future getPredictionPDF() async {
  var dir = await getApplicationDocumentsDirectory();
  File file = new File("${dir.path}/output.pdf");

  var url = 'http://10.0.2.2:5000/get-predictions';
  var uri = Uri.parse(url);

  var response = await http.get(uri);
  file.writeAsBytesSync(response.bodyBytes, flush: true);
  return file.path;
}
