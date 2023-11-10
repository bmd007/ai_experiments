package io.bmd007.ai.logexplorer;

import de.kherud.llama.InferenceParameters;
import de.kherud.llama.LlamaModel;
import de.kherud.llama.ModelParameters;
import lombok.extern.slf4j.Slf4j;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.boot.context.event.ApplicationReadyEvent;
import org.springframework.context.event.EventListener;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;

@Slf4j
@SpringBootApplication
public class LogExplorerApplication {

	public static void main(String[] args) {
		SpringApplication.run(LogExplorerApplication.class, args);
	}

	@EventListener(ApplicationReadyEvent.class)
	public void onApplicationReady() throws IOException {
//		LlamaModel.setLogger((level, message) -> System.out.print(message));
		ModelParameters modelParams = new ModelParameters.Builder()
				.setNGpuLayers(43)
				.build();
		InferenceParameters inferParams = new InferenceParameters.Builder()
				.setTemperature(0.7f)
				.setPenalizeNl(true)
				.setMirostat(InferenceParameters.MiroStat.V2)
				.setAntiPrompt("\n")
				.build();

		String modelPath = "/Users/mohami/workspacce/personal-repositories/ai_experiments/models/luna-ai-llama2";
		String system = "This is a conversation between User and Llama, a friendly chatbot.\n" +
				"Llama is helpful, kind, honest, good at writing, and never fails to answer any " +
				"requests immediately and with precision.\n";
		BufferedReader reader = new BufferedReader(new InputStreamReader(System.in, StandardCharsets.UTF_8));
		try (LlamaModel model = new LlamaModel(modelPath, modelParams)) {
			System.out.print(system);
			String prompt = system;
			while (true) {
				prompt += "\nUser: ";
				System.out.print("\nUser: ");
				String input = reader.readLine();
				prompt += input;
				System.out.print("Llama: ");
				prompt += "\nLlama: ";
				for (LlamaModel.Output output : model.generate(prompt, inferParams)) {
					System.out.print(output.text);
					prompt += output;
				}
			}
		}
	}

}
