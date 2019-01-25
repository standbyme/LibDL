package com.jstarcraft.core.codec.protocolbufferx.converter;

import java.util.EnumMap;

import com.jstarcraft.core.codec.protocolbufferx.ProtocolReference;
import com.jstarcraft.core.codec.specification.ClassDefinition;
import com.jstarcraft.core.codec.specification.CodecDefinition;
import com.jstarcraft.core.codec.specification.CodecSpecification;

/**
 * 协议上下文
 * 
 * @author Birdy
 *
 */
public abstract class ProtocolContext {

	protected static final EnumMap<CodecSpecification, ProtocolConverter<?>> converters = new EnumMap<>(CodecSpecification.class);
	static {
		converters.put(CodecSpecification.ARRAY, new ArrayConverter());
		converters.put(CodecSpecification.BOOLEAN, new BooleanConverter());
		converters.put(CodecSpecification.COLLECTION, new CollectionConverter());
		converters.put(CodecSpecification.ENUMERATION, new EnumerationConverter());
		converters.put(CodecSpecification.MAP, new MapConverter());
		converters.put(CodecSpecification.NUMBER, new NumberConverter());
		converters.put(CodecSpecification.OBJECT, new ObjectConverter());
		converters.put(CodecSpecification.STRING, new StringConverter());
		converters.put(CodecSpecification.TIME, new TimeConverter());
		converters.put(CodecSpecification.TYPE, new TypeConverter());
		converters.put(CodecSpecification.VOID, new VoidConverter());
	}

	/** 协议定义 */
	private final CodecDefinition definition;

	/** 读写上下文过程的数组引用 */
	protected ProtocolReference<Object> arrayReference = new ProtocolReference<Object>();
	/** 读写上下文过程的集合引用 */
	protected ProtocolReference<Object> collectionReference = new ProtocolReference<Object>();
	/** 读写上下文过程的映射引用 */
	protected ProtocolReference<Object> mapReference = new ProtocolReference<Object>();
	/** 读写上下文过程的对象引用 */
	protected ProtocolReference<Object> objectReference = new ProtocolReference<Object>();
	/** 读写上下文过程的字符串引用 */
	protected ProtocolReference<String> stringReference = new ProtocolReference<String>();

	public ProtocolContext(CodecDefinition definition) {
		this.definition = definition;
	}

	public ProtocolConverter getProtocolConverter(CodecSpecification specification) {
		ProtocolConverter converter = converters.get(specification);
		return converter;
	}

	protected ClassDefinition getClassDefinition(int index) {
		return definition.getClassDefinition(index);
	}

	protected ClassDefinition getClassDefinition(Class<?> clazz) {
		return definition.getClassDefinition(clazz);
	}

	protected Object getArrayValue(int index) {
		return arrayReference.getValue(index);
	}

	protected Integer getArrayIndex(Object value) {
		return arrayReference.getIndex(value);
	}

	protected int putArrayValue(Object value) {
		return arrayReference.putValue(value);
	}

	protected Object getCollectionValue(int index) {
		return collectionReference.getValue(index);
	}

	protected Integer getCollectionIndex(Object value) {
		return collectionReference.getIndex(value);
	}

	protected int putCollectionValue(Object value) {
		return collectionReference.putValue(value);
	}

	protected Object getMapValue(int index) {
		return mapReference.getValue(index);
	}

	protected Integer getMapIndex(Object value) {
		return mapReference.getIndex(value);
	}

	protected int putMapValue(Object value) {
		return mapReference.putValue(value);
	}

	protected Object getObjectValue(int index) {
		return objectReference.getValue(index);
	}

	protected Integer getObjectIndex(Object value) {
		return objectReference.getIndex(value);
	}

	protected int putObjectValue(Object value) {
		return objectReference.putValue(value);
	}

	protected String getStringValue(int index) {
		return stringReference.getValue(index);
	}

	protected Integer getStringIndex(String value) {
		return stringReference.getIndex(value);
	}

	protected int putStringValue(String value) {
		return stringReference.putValue(value);
	}

}
