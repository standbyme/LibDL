package com.jstarcraft.core.codec.json;

import com.fasterxml.jackson.databind.JavaType;

public class JsonContext {

	public static final String TYPE_FIELD = "type";

	public static final String CONTENT_FIELD = "content";

	private JavaType type;

	private Object content;

	public JavaType getType() {
		return type;
	}

	public Object getContent() {
		return content;
	}

	public static JsonContext instanceOf(JavaType type, Object content) {
		JsonContext instance = new JsonContext();
		instance.type = type;
		instance.content = content;
		return instance;
	}

}
