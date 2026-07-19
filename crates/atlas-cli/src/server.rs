use std::{
    io::{BufRead, BufReader, Read, Write},
    net::{TcpListener, TcpStream},
    path::Path,
    time::{SystemTime, UNIX_EPOCH},
};

use anyhow::{Context, Result, bail};
use atlas_model::AtlasModel;
use serde_json::{Value, json};

use crate::generate_completion;

pub(crate) fn serve(model_name: &str, directory: &Path, host: &str, port: u16) -> Result<()> {
    let listener =
        TcpListener::bind((host, port)).with_context(|| format!("bind http://{host}:{port}"))?;
    let model = AtlasModel::load(directory)?;
    eprintln!("atlas: serving {model_name} at http://{host}:{port} (one request at a time)");
    for stream in listener.incoming() {
        match stream {
            Ok(stream) => {
                if let Err(error) = handle_connection(stream, model_name, &model) {
                    eprintln!("atlas: request failed: {error:#}");
                }
            }
            Err(error) => eprintln!("atlas: accept failed: {error}"),
        }
    }
    Ok(())
}

fn handle_connection(mut stream: TcpStream, model_name: &str, model: &AtlasModel) -> Result<()> {
    let request = read_request(&mut stream)?;
    match (request.method.as_str(), request.path.as_str()) {
        ("GET", "/health") => {
            write_json(&mut stream, 200, &json!({"status":"ok","model":model_name}))
        }
        ("GET", "/v1/models") => write_json(
            &mut stream,
            200,
            &json!({"object":"list","data":[{"id":model_name,"object":"model"}]}),
        ),
        ("POST", "/v1/chat/completions") => {
            chat_completion(&mut stream, model_name, model, &request.body)
        }
        _ => write_json(
            &mut stream,
            404,
            &json!({"error":{"message":"not found","type":"invalid_request_error"}}),
        ),
    }
}

fn chat_completion(
    stream: &mut TcpStream,
    model_name: &str,
    model: &AtlasModel,
    body: &[u8],
) -> Result<()> {
    let body: Value = serde_json::from_slice(body).context("parse chat-completions JSON")?;
    let prompt = chat_prompt(&body)?;
    let max_tokens = body.get("max_tokens").and_then(Value::as_u64).unwrap_or(64) as usize;
    let stream_response = body.get("stream").and_then(Value::as_bool).unwrap_or(false);
    let result = generate_completion(model, &prompt, max_tokens)?;
    let id = format!(
        "chatcmpl-{}",
        SystemTime::now().duration_since(UNIX_EPOCH)?.as_millis()
    );
    if stream_response {
        write!(
            stream,
            "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n"
        )?;
        for token in &result.generation.generated_token_ids {
            let text = model.decode(&[*token])?;
            let chunk = json!({"id":id,"object":"chat.completion.chunk","model":model_name,"choices":[{"index":0,"delta":{"content":text},"finish_reason":Value::Null}]});
            write!(stream, "data: {}\n\n", serde_json::to_string(&chunk)?)?;
        }
        let finish = json!({"id":id,"object":"chat.completion.chunk","model":model_name,"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]});
        write!(
            stream,
            "data: {}\n\ndata: [DONE]\n\n",
            serde_json::to_string(&finish)?
        )?;
        return Ok(());
    }
    let completion = model.decode(&result.generation.generated_token_ids)?;
    write_json(
        stream,
        200,
        &json!({"id":id,"object":"chat.completion","model":model_name,"choices":[{"index":0,"message":{"role":"assistant","content":completion},"finish_reason":"stop"}],"usage":{"prompt_tokens":result.generation.prompt_token_ids.len(),"completion_tokens":result.generation.generated_token_ids.len(),"total_tokens":result.generation.prompt_token_ids.len()+result.generation.generated_token_ids.len()}}),
    )
}

struct HttpRequest {
    method: String,
    path: String,
    body: Vec<u8>,
}

fn read_request(stream: &mut TcpStream) -> Result<HttpRequest> {
    let mut reader = BufReader::new(stream.try_clone()?);
    let mut request_line = String::new();
    reader.read_line(&mut request_line)?;
    let mut parts = request_line.split_whitespace();
    let method = parts.next().context("missing HTTP method")?.to_owned();
    let path = parts
        .next()
        .context("missing HTTP path")?
        .split('?')
        .next()
        .unwrap_or("/")
        .to_owned();
    let mut content_length = 0usize;
    loop {
        let mut line = String::new();
        reader.read_line(&mut line)?;
        if line == "\r\n" || line.is_empty() {
            break;
        }
        if let Some(value) = line
            .strip_prefix("Content-Length:")
            .or_else(|| line.strip_prefix("content-length:"))
        {
            content_length = value.trim().parse().context("invalid Content-Length")?;
        }
    }
    if content_length > 1024 * 1024 {
        bail!("request body exceeds 1 MiB");
    }
    let mut body = vec![0; content_length];
    reader.read_exact(&mut body)?;
    Ok(HttpRequest { method, path, body })
}

fn chat_prompt(body: &Value) -> Result<String> {
    let messages = body
        .get("messages")
        .and_then(Value::as_array)
        .context("messages must be an array")?;
    let mut prompt = String::new();
    for message in messages {
        let role = message
            .get("role")
            .and_then(Value::as_str)
            .context("message role must be a string")?;
        let content = message
            .get("content")
            .and_then(Value::as_str)
            .context("message content must be a string")?;
        prompt.push_str(role);
        prompt.push_str(": ");
        prompt.push_str(content);
        prompt.push('\n');
    }
    if prompt.is_empty() {
        bail!("messages must not be empty");
    }
    Ok(prompt)
}

fn write_json(stream: &mut TcpStream, status: u16, body: &Value) -> Result<()> {
    let text = serde_json::to_string(body)?;
    let reason = match status {
        200 => "OK",
        404 => "Not Found",
        429 => "Too Many Requests",
        _ => "Bad Request",
    };
    write!(
        stream,
        "HTTP/1.1 {status} {reason}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{text}",
        text.len()
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn chat_prompt_preserves_message_order() {
        assert_eq!(chat_prompt(&json!({"messages":[{"role":"system","content":"brief"},{"role":"user","content":"hello"}]})).unwrap(), "system: brief\nuser: hello\n");
    }
}
