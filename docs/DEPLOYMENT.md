# Deploy BeatLedger

Use a public GitHub repository with the synthetic fixtures and the Streamlit Community Cloud entry point `app.py` on the default branch. Python 3.12 is the tested runtime. Publish the source and synthetic sample only. [Streamlit deployment instructions](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy)

## Server settings

In Community Cloud, set this root-level secret to enable hosted mode.

```toml
BEATLEDGER_DEPLOYMENT = "hosted"
```

## Use your own API key

In the hosted sidebar, choose Groq, Google Gemini, OpenAI or OpenRouter under **AI judge**. Keep the suggested model or enter a chat model ID that supports JSON mode. Paste the provider's key and click **Save key for this session**, then choose a source and click **Run AI analysis**. The key stays in that Streamlit session and can be reused after switching models or returning to that provider. **Remove saved key** clears it. A new session requires the key again.

Each run sends excerpts from up to 12 candidate cuts to the selected provider over HTTPS, using that provider's quota or credits. Turn off remote judging by selecting **Offline baseline**. Parsing, cut verification and exports still run in BeatLedger. The saved model run is always available on the bundled sample.

## Optional shared sample judge

With **Offline baseline** selected, **Judge sample live** uses the bundled synthetic scripts when a server-managed remote provider is configured. This shared key is used only for the bundled sample.

For an approved free-tier provider, add these settings directly in the app’s Secrets panel. Use the endpoint, model identifier, and response format supported by that provider.

```toml
BEATLEDGER_BASE_URL = "https://provider.example/v1"
BEATLEDGER_MODEL = "provider-model-id"
BEATLEDGER_API_KEY = "paste-the-key-into-Streamlit-Secrets"
BEATLEDGER_RESPONSE_FORMAT = "json"
BEATLEDGER_TIMEOUT_SECONDS = "15"
BEATLEDGER_MAX_OUTPUT_TOKENS = "500"
```

The shared sample judge evaluates at most four synthetic cuts per run and reuses the result for 15 minutes per app process. [Streamlit secrets documentation](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/secrets-management)

## Verify a published deployment

Open the deployment from a signed-out browser. Confirm that the bundled sample shows three episodes, that the saved model run has nine judged cuts, and that a fresh synthetic upload can produce downloadable CSV and JSON. Run AI analysis with a provider key and change models without re-entering it. Remove the key and check that the AI run button is disabled. Confirm that an approved silent cut requires footage confirmation and that a trim resets its status. Run the test and batch commands in the public README from a fresh checkout.
