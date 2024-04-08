package io.github.bmd007.experiment.ai.springailocally;

import lombok.extern.slf4j.Slf4j;
import org.springframework.ai.chat.messages.UserMessage;
import org.springframework.ai.chat.prompt.Prompt;
import org.springframework.ai.embedding.EmbeddingClient;
import org.springframework.ai.embedding.EmbeddingResponse;
import org.springframework.ai.ollama.OllamaChatClient;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.boot.context.event.ApplicationReadyEvent;
import org.springframework.context.event.EventListener;
import reactor.core.publisher.Flux;

import java.util.List;

@Slf4j
@SpringBootApplication
public class SpringAiLocallyApplication {

    public static void main(String[] args) {
        SpringApplication.run(SpringAiLocallyApplication.class, args);
    }

    @Autowired
    private OllamaChatClient chatClient;

    @Autowired
    private EmbeddingClient embeddingClient;

    @EventListener(ApplicationReadyEvent.class)
    public void onApplicationReady() {
        String message = "Hello";
        EmbeddingResponse embeddingResponse = embeddingClient.embedForResponse(List.of(message));
        Flux.fromIterable(embeddingResponse.getResult().getOutput())
                .subscribe(aDouble -> log.info("embedding {}", aDouble));

        log.info("embedding Hello {}", embeddingClient.embed("Hello"));

        Prompt prompt = new Prompt(new UserMessage(message));
        chatClient.stream(prompt)
                .flatMap(chatResponse -> Flux.fromIterable(chatResponse.getResults()))
                .map(chatResult -> chatResult.getOutput().getContent())
                .doOnNext(chatResult -> log.info("chatResult {}", chatResult))
                .subscribe();
    }
}
