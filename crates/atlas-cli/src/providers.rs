//! Provider-neutral model discovery and authentication primitives.
//!
//! Provider implementations own their wire protocol.  The CLI only deals in a
//! stable candidate ID and a resolved, immutable download plan.

use std::{
    collections::BTreeSet,
    env, fs,
    io::{Read, Write},
    path::{Path, PathBuf},
    thread,
    time::Duration,
};

use anyhow::{Context, Result, bail, ensure};
use keyring::Entry;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

pub const HUGGING_FACE: &str = "huggingface";
pub const SEARCH_PAGE_SIZE: usize = 15;

fn is_gated(metadata: &Value) -> bool {
    metadata
        .get("gated")
        .is_some_and(|value| !matches!(value, Value::Bool(false) | Value::Null))
        || metadata
            .get("private")
            .and_then(Value::as_bool)
            .unwrap_or(false)
}
const KEYCHAIN_SERVICE: &str = "dev.bisegni.atlas.model-provider";

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProviderSelection {
    Explicit(String),
    Default(String),
    Implicit(String),
}

impl ProviderSelection {
    pub fn id(&self) -> &str {
        match self {
            Self::Explicit(id) | Self::Default(id) | Self::Implicit(id) => id,
        }
    }
}

pub fn registered() -> &'static [&'static str] {
    &[HUGGING_FACE]
}

pub fn config_path() -> Result<PathBuf> {
    let home = env::var_os("HOME").context("HOME is required for provider preferences")?;
    Ok(PathBuf::from(home).join("Library/Application Support/atlas/providers.toml"))
}

pub fn load_default_provider() -> Result<Option<String>> {
    let path = config_path()?;
    let Ok(contents) = fs::read_to_string(&path) else {
        return Ok(None);
    };
    Ok(contents.lines().find_map(|line| {
        line.trim()
            .strip_prefix("default_provider = ")?
            .trim()
            .strip_prefix('"')?
            .strip_suffix('"')
            .map(str::to_owned)
    }))
}

pub fn set_default_provider(provider: Option<&str>) -> Result<()> {
    if let Some(provider) = provider {
        ensure!(
            registered().contains(&provider),
            "provider `{provider}` is not registered"
        );
    }
    let path = config_path()?;
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    if let Some(provider) = provider {
        let temporary = path.with_extension("toml.tmp");
        fs::write(&temporary, format!("default_provider = \"{provider}\"\n"))?;
        fs::rename(temporary, path)?;
    } else if path.exists() {
        fs::remove_file(path)?;
    }
    Ok(())
}

pub fn select(
    explicit: Option<&str>,
    configured: Option<&str>,
    registered: &[&str],
) -> Result<ProviderSelection> {
    if let Some(provider) = explicit {
        ensure!(
            registered.contains(&provider),
            "provider `{provider}` is not registered"
        );
        return Ok(ProviderSelection::Explicit(provider.to_owned()));
    }
    if let Some(provider) = configured {
        ensure!(
            registered.contains(&provider),
            "configured default provider `{provider}` is not registered; set a new default or clear it"
        );
        return Ok(ProviderSelection::Default(provider.to_owned()));
    }
    if let [provider] = registered {
        return Ok(ProviderSelection::Implicit((*provider).to_owned()));
    }
    bail!(
        "multiple model providers are registered ({}); pass --provider or run `atlas-cli provider default <provider>`",
        registered.join(", ")
    )
}

pub fn selected(explicit: Option<&str>) -> Result<ProviderSelection> {
    select(explicit, load_default_provider()?.as_deref(), registered())
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AuthSource {
    Environment,
    Keychain,
    Missing,
}

pub fn token(provider: &str) -> Result<(AuthSource, Option<String>)> {
    ensure!(
        registered().contains(&provider),
        "provider `{provider}` is not registered"
    );
    if let Some(value) = env::var_os("HF_TOKEN").filter(|value| !value.is_empty()) {
        return Ok((
            AuthSource::Environment,
            Some(value.to_string_lossy().into_owned()),
        ));
    }
    let entry = Entry::new(KEYCHAIN_SERVICE, provider).context("open provider keychain entry")?;
    match entry.get_password() {
        Ok(value) => Ok((AuthSource::Keychain, Some(value))),
        Err(keyring::Error::NoEntry) => Ok((AuthSource::Missing, None)),
        Err(error) => Err(error).context("read provider keychain entry"),
    }
}

pub fn store_token(provider: &str, value: &str) -> Result<()> {
    ensure!(
        registered().contains(&provider),
        "provider `{provider}` is not registered"
    );
    ensure!(
        value.starts_with("hf_"),
        "Hugging Face access tokens start with `hf_`"
    );
    Entry::new(KEYCHAIN_SERVICE, provider)?
        .set_password(value)
        .context("store provider token")
}

pub fn logout(provider: &str) -> Result<()> {
    ensure!(
        registered().contains(&provider),
        "provider `{provider}` is not registered"
    );
    let entry = Entry::new(KEYCHAIN_SERVICE, provider)?;
    match entry.delete_credential() {
        Ok(()) | Err(keyring::Error::NoEntry) => Ok(()),
        Err(error) => Err(error).context("delete provider keychain entry"),
    }
}

pub fn validate_hugging_face_token(value: &str) -> Result<()> {
    let response = ureq::get("https://huggingface.co/api/whoami-v2")
        .header("Authorization", format!("Bearer {value}"))
        .call()
        .context("validate Hugging Face token")?;
    ensure!(
        response.status().is_success(),
        "Hugging Face rejected the token"
    );
    Ok(())
}

pub trait ModelProvider {
    fn search(&self, query: &str, cursor: Option<&str>) -> Result<SearchPage>;
}

pub struct SearchPage {
    pub candidates: Vec<ModelCandidate>,
    pub next_cursor: Option<String>,
}

#[derive(Debug, Clone)]
pub struct ModelCandidate {
    pub provider: String,
    pub repository: String,
    pub revision: String,
    pub architecture: String,
    pub format: String,
    pub artifact: Option<String>,
    pub bytes: u64,
    pub requires_auth: bool,
    pub downloadable: bool,
    pub reason: Option<String>,
}

impl ModelCandidate {
    pub fn id(&self) -> String {
        match &self.artifact {
            Some(artifact) => format!(
                "{}:{}@{}:{}:{}",
                self.provider, self.repository, self.revision, self.format, artifact
            ),
            None => format!(
                "{}:{}@{}:{}",
                self.provider, self.repository, self.revision, self.format
            ),
        }
    }
    pub fn json(&self) -> Value {
        json!({"provider":self.provider,"provider_model_id":self.id(),"repository":self.repository,"revision":self.revision,"architecture":self.architecture,"format":self.format,"artifact":self.artifact,"bytes":self.bytes,"requires_auth":self.requires_auth,"downloadable":self.downloadable,"reason":self.reason})
    }
}

pub struct HuggingFaceProvider;

impl HuggingFaceProvider {
    fn get_json(url: &str) -> Result<Value> {
        let mut response = None;
        for attempt in 0..3 {
            match ureq::get(url).call() {
                Ok(value) => {
                    response = Some(value);
                    break;
                }
                Err(ureq::Error::StatusCode(429)) if attempt < 2 => {
                    thread::sleep(Duration::from_secs((attempt + 1) as u64))
                }
                Err(error) => {
                    return Err(error)
                        .with_context(|| format!("Hugging Face request failed: {url}"));
                }
            }
        }
        let response = response
            .context("Hugging Face rate limit exceeded; wait a moment and retry the search")?;
        let mut body = response.into_body();
        serde_json::from_str(
            &body
                .read_to_string()
                .context("read Hugging Face response")?,
        )
        .context("parse Hugging Face response")
    }

    fn search_json(url: &str) -> Result<(Value, Option<String>)> {
        let response = ureq::get(url)
            .call()
            .with_context(|| format!("Hugging Face request failed: {url}"))?;
        let next = response
            .headers()
            .get("link")
            .and_then(|value| value.to_str().ok())
            .and_then(|value| value.split("cursor=").nth(1))
            .and_then(|value| value.split('>').next())
            .map(str::to_owned);
        let mut body = response.into_body();
        Ok((serde_json::from_str(&body.read_to_string()?)?, next))
    }

    fn candidate(repository: &str) -> Result<Vec<ModelCandidate>> {
        let metadata = Self::get_json(&format!(
            "https://huggingface.co/api/models/{repository}?blobs=true"
        ))?;
        Self::candidate_from_metadata(repository, &metadata)
    }

    fn candidate_from_metadata(repository: &str, metadata: &Value) -> Result<Vec<ModelCandidate>> {
        let revision = metadata
            .get("sha")
            .and_then(Value::as_str)
            .context("Hugging Face model is missing immutable sha")?;
        let config_architecture = metadata
            .get("config")
            .and_then(|v| v.get("architectures"))
            .and_then(Value::as_array)
            .and_then(|v| v.first())
            .and_then(Value::as_str);
        let gguf_architecture = metadata
            .get("gguf")
            .and_then(|value| value.get("architecture"))
            .and_then(Value::as_str);
        let llama = config_architecture == Some("LlamaForCausalLM");
        let gemma4 = gguf_architecture == Some("gemma4");
        if !llama && !gemma4 {
            return Ok(Vec::new());
        }
        let siblings = metadata
            .get("siblings")
            .and_then(Value::as_array)
            .context("Hugging Face model is missing files")?;
        let files: Vec<&str> = siblings
            .iter()
            .filter_map(|v| v.get("rfilename").and_then(Value::as_str))
            .collect();
        if llama && (!files.contains(&"config.json") || !files.contains(&"tokenizer.json")) {
            return Ok(Vec::new());
        }
        let safe = files
            .iter()
            .filter(|name| name.ends_with(".safetensors"))
            .count();
        let gguf = files.iter().filter(|name| name.ends_with(".gguf")).count();
        if files
            .iter()
            .any(|name| name.contains("model") && name.ends_with(".bin"))
        {
            return Ok(Vec::new());
        }
        let formats: Vec<(&str, Option<&str>)> = if safe > 0 && gguf == 0 {
            vec![("safetensors-fp32", None)]
        } else if safe == 0 {
            files
                .iter()
                .filter_map(|file| {
                    if gemma4
                        && file.ends_with(".gguf")
                        && file.contains("q4_0")
                        && !file.contains("mmproj")
                    {
                        Some(("gguf-gemma4-q4_0", Some(*file)))
                    } else if file.ends_with(".gguf") && file.contains("Q4_0") {
                        Some(("gguf-q4_0", Some(*file)))
                    } else if file.ends_with(".gguf") && file.contains("Q8_0") {
                        Some(("gguf-q8_0", Some(*file)))
                    } else {
                        None
                    }
                })
                .collect()
        } else {
            Vec::new()
        };
        if formats.is_empty() {
            return Ok(Vec::new());
        }
        let required: BTreeSet<&str> = files
            .iter()
            .copied()
            .filter(|name| {
                matches!(
                    *name,
                    "config.json" | "tokenizer.json" | "model.safetensors.index.json"
                ) || name.ends_with(".safetensors")
                    || name.ends_with(".gguf")
            })
            .collect();
        let bytes = siblings
            .iter()
            .filter(|file| {
                file.get("rfilename")
                    .and_then(Value::as_str)
                    .is_some_and(|name| required.contains(name))
            })
            .filter_map(|v| {
                v.get("size").and_then(Value::as_u64).or_else(|| {
                    v.get("lfs")
                        .and_then(|lfs| lfs.get("size"))
                        .and_then(Value::as_u64)
                })
            })
            .sum();
        Ok(formats
            .into_iter()
            .map(|(format, artifact)| ModelCandidate {
                provider: HUGGING_FACE.into(),
                repository: repository.into(),
                revision: revision.into(),
                architecture: if gemma4 {
                    "gemma4".into()
                } else {
                    "LlamaForCausalLM".into()
                },
                format: format.into(),
                artifact: artifact.map(str::to_owned),
                bytes: artifact
                    .and_then(|name| {
                        siblings
                            .iter()
                            .find(|value| {
                                value.get("rfilename").and_then(Value::as_str) == Some(name)
                            })
                            .and_then(|value| value.get("lfs"))
                            .and_then(|lfs| lfs.get("size"))
                            .and_then(Value::as_u64)
                    })
                    .unwrap_or(bytes),
                requires_auth: is_gated(&metadata),
                downloadable: true,
                reason: None,
            })
            .collect())
    }
}

#[derive(Debug)]
pub struct DownloadedModel {
    pub repository: String,
    pub revision: String,
    pub files: Vec<String>,
}

pub fn download_hugging_face(
    candidate_id: &str,
    destination: &Path,
    allow_auth: bool,
) -> Result<DownloadedModel> {
    let encoded = candidate_id
        .strip_prefix("huggingface:")
        .context("download requires a Hugging Face candidate ID")?;
    let (repository, remainder) = encoded
        .rsplit_once('@')
        .context("invalid provider model ID")?;
    let (revision, format_and_artifact) = remainder
        .split_once(':')
        .context("invalid provider model ID")?;
    let (advertised_format, selected_artifact) = format_and_artifact
        .split_once(':')
        .map_or((format_and_artifact, None), |(format, artifact)| {
            (format, Some(artifact))
        });
    ensure!(
        revision.len() >= 7 && revision.bytes().all(|byte| byte.is_ascii_hexdigit()),
        "provider model ID must use an immutable revision"
    );
    let metadata = HuggingFaceProvider::get_json(&format!(
        "https://huggingface.co/api/models/{repository}/revision/{revision}?blobs=true"
    ))?;
    let config_architecture = metadata
        .get("config")
        .and_then(|v| v.get("architectures"))
        .and_then(Value::as_array)
        .and_then(|v| v.first())
        .and_then(Value::as_str);
    let gguf_architecture = metadata
        .get("gguf")
        .and_then(|value| value.get("architecture"))
        .and_then(Value::as_str);
    let llama = config_architecture == Some("LlamaForCausalLM");
    let gemma4 = gguf_architecture == Some("gemma4");
    ensure!(
        llama || gemma4,
        "provider artifact architecture is unsupported"
    );
    let gated = is_gated(&metadata);
    let access_token = if gated && allow_auth {
        let (_, token) = token(HUGGING_FACE)?;
        ensure!(
            token.is_some(),
            "Hugging Face credentials are required for this gated/private artifact; run `atlas-cli provider login huggingface` or set HF_TOKEN"
        );
        token
    } else if gated {
        bail!("Hugging Face credentials are required for this gated/private artifact")
    } else {
        None
    };
    let siblings = metadata
        .get("siblings")
        .and_then(Value::as_array)
        .context("Hugging Face model is missing files")?;
    let all: Vec<String> = siblings
        .iter()
        .filter_map(|v| v.get("rfilename").and_then(Value::as_str))
        .map(str::to_owned)
        .collect();
    let mut files = if llama {
        vec!["config.json".to_owned(), "tokenizer.json".to_owned()]
    } else {
        Vec::new()
    };
    match advertised_format {
        "safetensors-fp32" => {
            let weights: Vec<_> = all
                .iter()
                .filter(|name| name.ends_with(".safetensors"))
                .cloned()
                .collect();
            ensure!(
                !weights.is_empty(),
                "resolved repository has no SafeTensors artifacts"
            );
            files.extend(weights);
            if all
                .iter()
                .any(|name| name == "model.safetensors.index.json")
            {
                files.push("model.safetensors.index.json".into());
            }
        }
        "gguf-q4_0" | "gguf-q8_0" | "gguf-gemma4-q4_0" => {
            ensure!(
                advertised_format != "gguf-gemma4-q4_0" || gemma4,
                "Gemma GGUF format requires gemma4 architecture"
            );
            let weights: Vec<_> = all
                .iter()
                .filter(|name| name.ends_with(".gguf"))
                .cloned()
                .collect();
            let artifact =
                selected_artifact.context("GGUF provider model ID is missing its artifact name")?;
            ensure!(
                weights.iter().any(|weight| weight == artifact),
                "selected GGUF artifact no longer exists at the pinned revision"
            );
            files.push(artifact.to_owned());
        }
        _ => bail!("unsupported provider artifact format `{advertised_format}`"),
    }
    ensure!(
        files
            .iter()
            .all(|name| !name.contains('/') && !name.contains('\\')),
        "provider returned an unsafe artifact path"
    );
    fs::create_dir_all(destination)?;
    for name in &files {
        let url = format!("https://huggingface.co/{repository}/resolve/{revision}/{name}");
        let mut request = ureq::get(&url);
        if let Some(token) = &access_token {
            request = request.header("Authorization", format!("Bearer {token}"));
        }
        let response = request
            .call()
            .with_context(|| format!("download {repository}/{name}"))?;
        let mut input = response.into_body().into_reader();
        let path = destination.join(name);
        let mut output = fs::File::create(&path)?;
        let mut hash = Sha256::new();
        let mut buffer = [0u8; 64 * 1024];
        loop {
            let read = input.read(&mut buffer)?;
            if read == 0 {
                break;
            }
            hash.update(&buffer[..read]);
            output.write_all(&buffer[..read])?;
        }
        ensure!(
            fs::metadata(&path)?.len() > 0,
            "downloaded file is empty: {name}"
        );
        let _digest = format!("{:x}", hash.finalize()); // manifest registration records the digest.
    }
    Ok(DownloadedModel {
        repository: repository.into(),
        revision: revision.into(),
        files,
    })
}

impl ModelProvider for HuggingFaceProvider {
    fn search(&self, query: &str, cursor: Option<&str>) -> Result<SearchPage> {
        ensure!(
            !query.trim().is_empty(),
            "model search query may not be empty"
        );
        let encoded = query.replace(' ', "%20");
        let url = match cursor {
            Some(cursor) => format!(
                "https://huggingface.co/api/models?search={encoded}&limit={SEARCH_PAGE_SIZE}&cursor={cursor}"
            ),
            None => format!(
                "https://huggingface.co/api/models?search={encoded}&limit={SEARCH_PAGE_SIZE}"
            ),
        };
        let (response, next_cursor) = Self::search_json(&url)?;
        let mut candidates = Vec::new();
        for item in response
            .as_array()
            .context("Hugging Face search response is not an array")?
        {
            if let Some(id) = item.get("id").and_then(Value::as_str) {
                let compatible = Self::candidate(id)?;
                if compatible.is_empty() {
                    candidates.push(ModelCandidate { provider: HUGGING_FACE.into(), repository: id.into(), revision: "unknown".into(), architecture: "unsupported".into(), format: "unsupported".into(), artifact: None, bytes: 0, requires_auth: false, downloadable: false, reason: Some("not a complete Atlas-supported Llama artifact or Gemma 4 E2B Q4_0 GGUF".into()) });
                } else {
                    candidates.extend(compatible);
                }
            }
        }
        Ok(SearchPage {
            candidates,
            next_cursor,
        })
    }
}

pub fn provider(id: &str) -> Result<Box<dyn ModelProvider>> {
    match id {
        HUGGING_FACE => Ok(Box::new(HuggingFaceProvider)),
        _ => bail!("provider `{id}` is not registered"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn selection_prefers_explicit_then_default_then_single_provider() {
        assert_eq!(
            select(Some("huggingface"), None, &["huggingface"]).unwrap(),
            ProviderSelection::Explicit("huggingface".into())
        );
        assert_eq!(
            select(None, Some("huggingface"), &["huggingface", "other"]).unwrap(),
            ProviderSelection::Default("huggingface".into())
        );
        assert_eq!(
            select(None, None, &["huggingface"]).unwrap(),
            ProviderSelection::Implicit("huggingface".into())
        );
    }
    #[test]
    fn selection_rejects_ambiguous_or_invalid_provider() {
        assert!(select(None, None, &["huggingface", "other"]).is_err());
        assert!(select(None, Some("missing"), &["huggingface"]).is_err());
        assert!(select(Some("missing"), None, &["huggingface"]).is_err());
    }
    #[test]
    fn candidate_id_is_stable_and_provider_scoped() {
        let candidate = ModelCandidate {
            provider: HUGGING_FACE.into(),
            repository: "org/model".into(),
            revision: "abc".into(),
            architecture: "LlamaForCausalLM".into(),
            format: "safetensors-fp32".into(),
            artifact: None,
            bytes: 7,
            requires_auth: false,
            downloadable: true,
            reason: None,
        };
        assert_eq!(candidate.id(), "huggingface:org/model@abc:safetensors-fp32");
        assert_eq!(candidate.json()["provider"], "huggingface");
    }

    #[test]
    fn manual_gated_metadata_requires_authentication() {
        assert!(is_gated(&json!({"gated": "manual"})));
        assert!(is_gated(&json!({"private": true})));
        assert!(!is_gated(&json!({"gated": false})));
    }

    #[test]
    fn metadata_fixture_keeps_only_supported_complete_llama_artifacts() {
        let supported = json!({
            "sha": "0123456789abcdef",
            "config": {"architectures": ["LlamaForCausalLM"]},
            "siblings": [
                {"rfilename": "config.json", "size": 4},
                {"rfilename": "tokenizer.json", "size": 5},
                {"rfilename": "model.safetensors", "lfs": {"size": 6}}
            ]
        });
        let unsupported = json!({
            "sha": "0123456789abcdef",
            "config": {"architectures": ["MistralForCausalLM"]},
            "siblings": [
                {"rfilename": "config.json"},
                {"rfilename": "tokenizer.json"},
                {"rfilename": "model.safetensors"}
            ]
        });
        let mixed = json!({
            "sha": "0123456789abcdef",
            "config": {"architectures": ["LlamaForCausalLM"]},
            "siblings": [
                {"rfilename": "config.json"},
                {"rfilename": "tokenizer.json"},
                {"rfilename": "model.safetensors"},
                {"rfilename": "model-Q4_0.gguf"}
            ]
        });

        let candidates =
            HuggingFaceProvider::candidate_from_metadata("org/model", &supported).unwrap();
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].bytes, 15);
        assert!(
            HuggingFaceProvider::candidate_from_metadata("org/model", &unsupported)
                .unwrap()
                .is_empty()
        );
        assert!(
            HuggingFaceProvider::candidate_from_metadata("org/model", &mixed)
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn metadata_fixture_exposes_gemma4_text_gguf_without_mmproj() {
        let metadata = json!({
            "sha": "675cff42a74c774d6cb76f76d8eacb49b48c9b93",
            "config": {},
            "gguf": {"architecture": "gemma4"},
            "siblings": [
                {"rfilename": "gemma-4-E2B-it-mmproj.gguf", "lfs": {"size": 10}},
                {"rfilename": "gemma-4-E2B_q4_0-it.gguf", "lfs": {"size": 20}}
            ]
        });
        let candidates = HuggingFaceProvider::candidate_from_metadata(
            "google/gemma-4-E2B-it-qat-q4_0-gguf",
            &metadata,
        )
        .unwrap();
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].architecture, "gemma4");
        assert_eq!(candidates[0].format, "gguf-gemma4-q4_0");
        assert_eq!(
            candidates[0].artifact.as_deref(),
            Some("gemma-4-E2B_q4_0-it.gguf")
        );
        assert_eq!(candidates[0].bytes, 20);
    }
}
