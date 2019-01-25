package com.jstarcraft.core.codec.schema;

import org.springframework.beans.factory.xml.NamespaceHandlerSupport;

import com.jstarcraft.core.codec.schema.CodecXmlParser.ElementDefinition;

/**
 * 协议XML处理器
 * 
 * @author Birdy
 */
public class CodecXmlHandler extends NamespaceHandlerSupport {

	@Override
	public void init() {
		registerBeanDefinitionParser(ElementDefinition.CONFIGURATION.getName(), new CodecXmlParser());
	}

}
