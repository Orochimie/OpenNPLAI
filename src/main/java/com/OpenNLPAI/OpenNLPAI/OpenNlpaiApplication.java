package com.OpenNLPAI.OpenNLPAI;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.nio.charset.StandardCharsets;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

import opennlp.tools.langdetect.Language;
import opennlp.tools.langdetect.LanguageDetector;
import opennlp.tools.langdetect.LanguageDetectorFactory;
import opennlp.tools.langdetect.LanguageDetectorME;
import opennlp.tools.langdetect.LanguageDetectorModel;
import opennlp.tools.langdetect.LanguageDetectorSampleStream;
import opennlp.tools.langdetect.LanguageSample;
import opennlp.tools.ml.perceptron.PerceptronTrainer;
import opennlp.tools.util.InputStreamFactory;
import opennlp.tools.util.MarkableFileInputStreamFactory;
import opennlp.tools.util.ObjectStream;
import opennlp.tools.util.PlainTextByLineStream;
import opennlp.tools.util.TrainingParameters;
import opennlp.tools.util.model.ModelUtil;

@SpringBootApplication
public class OpenNlpaiApplication {

	public static void main(String[] args) throws IOException {
		SpringApplication.run(OpenNlpaiApplication.class, args);
		InputStreamFactory inputStreamFactory = new MarkableFileInputStreamFactory(new File("./eng_news_2016_100K-sentences.txt"));

		ObjectStream<String> lineStream = new PlainTextByLineStream(inputStreamFactory, StandardCharsets.UTF_8);
		ObjectStream<LanguageSample> sampleStream = new LanguageDetectorSampleStream(lineStream);

		TrainingParameters params = ModelUtil.createDefaultTrainingParameters();
		params.put(TrainingParameters.ALGORITHM_PARAM, PerceptronTrainer.PERCEPTRON_VALUE);
		params.put(TrainingParameters.CUTOFF_PARAM, 0);

		LanguageDetectorFactory factory = new LanguageDetectorFactory();

		LanguageDetectorModel model = LanguageDetectorME.train(sampleStream, params, factory);
		model.serialize(new File("./langdetect-183.bin"));
		LanguageDetector detector=new LanguageDetectorME(model);
		String inputText = "Hello";
		

		// Get the most probable language
		Language bestLanguage = detector.predictLanguage(inputText);
		System.out.println("Best language: " + bestLanguage.getLang());
		System.out.println("Best language confidence: " + bestLanguage.getConfidence());

		// Get an array with the most probable languages
		Language[] languages = detector.predictLanguages(null);

	}

}
