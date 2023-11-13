package io.bmd007.ai.logexplorer;

import com.hexadevlabs.gpt4all.LLModel;
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
import java.nio.file.Path;

@Slf4j
@SpringBootApplication
public class LogExplorerApplication {

    private static final String LUNA_AI_LLAMA2_Path = "/Users/mohami/workspacce/personal-repositories/ai_experiments/models/luna-ai-llama2";
    private static final String DOLPHON_MISTRAL_Path = "/Users/mohami/workspacce/personal-repositories/ai_experiments/models/dolphin-2.2.1-mistral-7b.Q8_0.gguf";
    private static final String MISTRAL_OPENORCA_PATH = "/Users/mohami/workspacce/personal-repositories/ai_experiments/models/mistral-7b-openorca.Q4_0.gguf";

    public static void main(String[] args) {
        SpringApplication.run(LogExplorerApplication.class, args);
    }

    @EventListener(ApplicationReadyEvent.class)
    public void onApplicationReady() {
//        gpt4allBridge(MISTRAL_OPENORCA_PATH);
        llamaCPPBridge(LUNA_AI_LLAMA2_Path);
    }

    void gpt4allBridge(String modelPath) {
        String prompt = "### Human:\nWhat is the meaning of life\n### Assistant:";

        try (LLModel model = new LLModel(Path.of(modelPath))) {

            LLModel.GenerationConfig config = LLModel.config()
                    .withNPredict(4096)
                    .withTemp(0)
                    .build();

            String fullGeneration = model.generate(prompt, config, true);

        } catch (Exception e) {
            // Exceptions generally may happen if the model file fails to load
            // for a number of reasons such as a file not found.
            // It is possible that Java may not be able to dynamically load the native shared library or
            // the llmodel shared library may not be able to dynamically load the backend
            // implementation for the model file you provided.
            //
            // Once the LLModel class is successfully loaded into memory the text generation calls
            // generally should not throw exceptions.
            e.printStackTrace(); // Printing here but in a production system you may want to take some action.
        }

    }

    void llamaCPPBridge(String modelPath) {
//		LlamaModel.setLogger((level, message) -> System.out.print(message));
        ModelParameters modelParams = new ModelParameters.Builder()
                .setNGpuLayers(43)
                .build();
        InferenceParameters inferParams = new InferenceParameters.Builder()
                .setTemperature(0f)
                .setPenalizeNl(true)
                .setMirostat(InferenceParameters.MiroStat.V2)
                .setAntiPrompt("\n")
                .build();

        String system = """
                This is a conversation between a human (user) and you (Llama), a friendly chatbot.
                Llama is helpful, kind, honest, good at writing, and never fails to answer any requests immediately and with precision.
                If you don't know the answer, respond by a request for search like following:
                {
                    "messageType": "searchRequest",
                    "value": ${what you don't know}
                }
                """;
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
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }
}
