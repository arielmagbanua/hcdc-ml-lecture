import 'package:flutter/material.dart';
import 'package:google_mlkit_translation/google_mlkit_translation.dart';

import '../activity_indicator/activity_indicator.dart';

class LanguageTranslatorView extends StatefulWidget {
  const LanguageTranslatorView({super.key});

  @override
  State<LanguageTranslatorView> createState() => _LanguageTranslatorViewState();
}

class _LanguageTranslatorViewState extends State<LanguageTranslatorView> {
  String? _translatedText;
  final _controller = TextEditingController();
  final _modelManager = OnDeviceTranslatorModelManager();
  final _sourceLanguage = TranslateLanguage.english;
  final _targetLanguage = TranslateLanguage.spanish;
  late final _onDeviceTranslator = OnDeviceTranslator(
    sourceLanguage: _sourceLanguage,
    targetLanguage: _targetLanguage,
  );

  @override
  void dispose() {
    _onDeviceTranslator.close();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return SafeArea(
      child: Scaffold(
        appBar: AppBar(
          title: const Text('On-device Translation'),
        ),
        body: GestureDetector(
          onTap: () {
            FocusScope.of(context).unfocus();
          },
          child: ListView(
            children: [
              const SizedBox(height: 30),
              Center(
                child: Text('Enter text (source: ${_sourceLanguage.name})'),
              ),
              Padding(
                padding: const EdgeInsets.all(20.0),
                child: Container(
                  padding: const EdgeInsets.symmetric(horizontal: 20),
                  decoration: BoxDecoration(
                    border: Border.all(
                      width: 2,
                    ),
                  ),
                  child: TextField(
                    controller: _controller,
                    decoration: const InputDecoration(border: InputBorder.none),
                    maxLines: null,
                  ),
                ),
              ),
              Center(
                child:
                    Text('Translated Text (target: ${_targetLanguage.name})'),
              ),
              const SizedBox(height: 30),
              Padding(
                padding: const EdgeInsets.all(20.0),
                child: Container(
                  width: MediaQuery.of(context).size.width / 1.3,
                  padding: const EdgeInsets.all(20),
                  decoration: BoxDecoration(
                    border: Border.all(
                      width: 2,
                    ),
                  ),
                  child: Text(_translatedText ?? ''),
                ),
              ),
              const SizedBox(height: 30),
              Row(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  ElevatedButton(
                    onPressed: _translateText,
                    child: const Text('Translate'),
                  )
                ],
              ),
              const SizedBox(height: 20),
              Row(
                mainAxisAlignment: MainAxisAlignment.spaceAround,
                children: [
                  ElevatedButton(
                    onPressed: _downloadSourceModel,
                    child: const Text('Download Source Model'),
                  ),
                  ElevatedButton(
                    onPressed: _downloadTargetModel,
                    child: const Text('Download Target Model'),
                  ),
                ],
              ),
              const SizedBox(height: 20),
              Row(
                mainAxisAlignment: MainAxisAlignment.spaceAround,
                children: [
                  ElevatedButton(
                    onPressed: _deleteSourceModel,
                    child: const Text('Delete Source Model'),
                  ),
                  ElevatedButton(
                    onPressed: _deleteTargetModel,
                    child: const Text('Delete Target Model'),
                  ),
                ],
              ),
              const SizedBox(height: 20),
              Row(
                mainAxisAlignment: MainAxisAlignment.spaceAround,
                children: [
                  ElevatedButton(
                    onPressed: _isSourceModelDownloaded,
                    child: const Text('Source Downloaded?'),
                  ),
                  ElevatedButton(
                    onPressed: _isTargetModelDownloaded,
                    child: const Text('Target Downloaded?'),
                  ),
                ],
              ),
            ],
          ),
        ),
      ),
    );
  }

  Future<void> _downloadSourceModel() async {
    Toast().show(
      'Downloading model (${_sourceLanguage.name})...',
      _modelManager
          .downloadModel(_sourceLanguage.bcpCode)
          .then((value) => value ? 'success' : 'failed'),
      context,
      this,
    );
  }

  Future<void> _downloadTargetModel() async {
    Toast().show(
      'Downloading model (${_targetLanguage.name})...',
      _modelManager
          .downloadModel(_targetLanguage.bcpCode)
          .then((value) => value ? 'success' : 'failed'),
      context,
      this,
    );
  }

  Future<void> _deleteSourceModel() async {
    Toast().show(
      'Deleting model (${_sourceLanguage.name})...',
      _modelManager
          .deleteModel(_sourceLanguage.bcpCode)
          .then((value) => value ? 'success' : 'failed'),
      context,
      this,
    );
  }

  Future<void> _deleteTargetModel() async {
    Toast().show(
      'Deleting model (${_targetLanguage.name})...',
      _modelManager
          .deleteModel(_targetLanguage.bcpCode)
          .then((value) => value ? 'success' : 'failed'),
      context,
      this,
    );
  }

  Future<void> _isSourceModelDownloaded() async {
    Toast().show(
      'Checking if model (${_sourceLanguage.name}) is downloaded...',
      _modelManager
          .isModelDownloaded(_sourceLanguage.bcpCode)
          .then((value) => value ? 'downloaded' : 'not downloaded'),
      context,
      this,
    );
  }

  Future<void> _isTargetModelDownloaded() async {
    Toast().show(
      'Checking if model (${_targetLanguage.name}) is downloaded...',
      _modelManager
          .isModelDownloaded(_targetLanguage.bcpCode)
          .then((value) => value ? 'downloaded' : 'not downloaded'),
      context,
      this,
    );
  }

  Future<void> _translateText() async {
    FocusScope.of(context).unfocus();
    final result = await _onDeviceTranslator.translateText(_controller.text);
    setState(() {
      _translatedText = result;
    });
  }
}
