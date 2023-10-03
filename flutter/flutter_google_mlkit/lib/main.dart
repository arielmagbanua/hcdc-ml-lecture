import 'package:camera/camera.dart';
import 'package:flutter/material.dart';

import 'nlp_detector_views/entity_extraction_view.dart';
import 'nlp_detector_views/language_identifier_view.dart';
import 'nlp_detector_views/language_translator_view.dart';
import 'nlp_detector_views/smart_reply_view.dart';
import 'vision_detector_views/barcode_scanner_view.dart';
import 'vision_detector_views/digital_ink_recognizer_view.dart';
import 'vision_detector_views/face_detector_view.dart';
import 'vision_detector_views/label_detector_view.dart';
import 'vision_detector_views/object_detector_view.dart';
import 'vision_detector_views/pose_detector_view.dart';
import 'vision_detector_views/selfie_segmenter_view.dart';
import 'vision_detector_views/text_detector_view.dart';

List<CameraDescription> cameras = [];

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();

  cameras = await availableCameras();

  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return const MaterialApp(
      debugShowCheckedModeBanner: false,
      home: Home(),
    );
  }
}

class Home extends StatelessWidget {
  const Home({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Google ML Kit Demo App'),
        centerTitle: true,
        elevation: 0,
      ),
      body: SafeArea(
        child: Center(
          child: SingleChildScrollView(
            child: Padding(
              padding: const EdgeInsets.symmetric(horizontal: 16),
              child: Column(
                children: [
                  ExpansionTile(
                    title: const Text('Vision APIs'),
                    children: [
                      DemoCard(
                        'Barcode Scanning',
                        BarcodeScannerView(),
                      ),
                      DemoCard(
                        'Face Detection',
                        FaceDetectorView(),
                      ),
                      DemoCard(
                        'Image Labeling',
                        ImageLabelView(),
                      ),
                      DemoCard(
                        'Object Detection',
                        ObjectDetectorView(),
                      ),
                      DemoCard(
                        'Text Recognition',
                        TextRecognizerView(),
                      ),
                      DemoCard(
                        'Digital Ink Recognition',
                        DigitalInkView(),
                      ),
                      DemoCard(
                        'Pose Detection',
                        PoseDetectorView(),
                      ),
                      DemoCard(
                        'Selfie Segmentation',
                        SelfieSegmenterView(),
                      ),
                    ],
                  ),
                  const SizedBox(height: 20),
                  const ExpansionTile(
                    title: Text('Natural Language APIs'),
                    children: [
                      DemoCard(
                        'Language ID',
                        LanguageIdentifierView(),
                      ),
                      DemoCard(
                        'On-device Translation',
                        LanguageTranslatorView(),
                      ),
                      DemoCard(
                        'Smart Reply',
                        SmartReplyView(),
                      ),
                      DemoCard(
                        'Entity Extraction',
                        EntityExtractionView(),
                      ),
                    ],
                  ),
                ],
              ),
            ),
          ),
        ),
      ),
    );
  }
}

class DemoCard extends StatelessWidget {
  final String _label;
  final Widget _viewPage;
  final bool featureCompleted;

  const DemoCard(
    this._label,
    this._viewPage, {
    super.key,
    this.featureCompleted = true,
  });

  @override
  Widget build(BuildContext context) {
    return Card(
      elevation: 5,
      margin: const EdgeInsets.only(bottom: 10),
      child: ListTile(
        tileColor: Theme.of(context).primaryColor,
        title: Text(
          _label,
          style: const TextStyle(
            color: Colors.white,
            fontWeight: FontWeight.bold,
          ),
        ),
        onTap: () {
          if (!featureCompleted) {
            ScaffoldMessenger.of(context).showSnackBar(
              const SnackBar(
                content: Text('This feature has not been implemented yet'),
              ),
            );
          } else {
            Navigator.push(
              context,
              MaterialPageRoute(builder: (context) => _viewPage),
            );
          }
        },
      ),
    );
  }
}
