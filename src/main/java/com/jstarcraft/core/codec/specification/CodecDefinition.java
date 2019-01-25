package com.jstarcraft.core.codec.specification;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.lang.reflect.GenericArrayType;
import java.lang.reflect.ParameterizedType;
import java.lang.reflect.Type;
import java.lang.reflect.TypeVariable;
import java.lang.reflect.WildcardType;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.TreeSet;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

import org.apache.commons.lang3.builder.EqualsBuilder;
import org.apache.commons.lang3.builder.HashCodeBuilder;

import com.jstarcraft.core.codec.protocolbufferx.exception.ProtocolDefinitionException;
import com.jstarcraft.core.utility.ClassUtility;
import com.jstarcraft.core.utility.PressUtility;

/**
 * 编解码定义(核心类)
 * 
 * @author Birdy
 */
public class CodecDefinition {

	private static final Comparator<Type> typeComparator = new Comparator<Type>() {
		@Override
		public int compare(Type left, Type right) {
			return left.getTypeName().compareTo(right.getTypeName());
		}
	};

	/** 代号管理器 */
	private AtomicInteger codeManager = new AtomicInteger();
	/** 索引-定义映射 */
	private ArrayList<ClassDefinition> code2Definitions = new ArrayList<>();
	/** 类型-定义映射 */
	private HashMap<Type, ClassDefinition> type2Definitions = new HashMap<>();

	// /** 索引-类型映射 */
	// private DualHashBidiMap simples = new DualHashBidiMap();

	private CodecDefinition() {
	}

	/**
	 * 获取类型定义
	 * 
	 * @param index
	 * @return
	 */
	public Collection<ClassDefinition> getClassDefinitions() {
		return Collections.unmodifiableCollection(code2Definitions);
	}

	/**
	 * 获取类型定义
	 * 
	 * @param index
	 * @return
	 */
	public ClassDefinition getClassDefinition(int index) {
		return code2Definitions.get(index - 1);
	}

	/**
	 * 获取类型定义
	 * 
	 * @param clazz
	 * @return
	 */
	public ClassDefinition getClassDefinition(Class<?> clazz) {
		while (clazz.isArray()) {
			clazz = clazz.getComponentType();
		}
		ClassDefinition definition = type2Definitions.get(clazz);
		if (definition != null) {
			// 存在类型映射
			return definition;
		}
		synchronized (this) {
			// 兼容类型
			// for (Class<?> type : clazz.getInterfaces()) {
			// definition = type2Definitions.get(type);
			// if (definition != null) {
			// type2Definitions.put(clazz, definition);
			// return definition;
			// }
			// }
			Class<?> type = clazz;
			while (type != Object.class) {
				type = type.getSuperclass();
				if (type == Object.class) {
					throw new ProtocolDefinitionException(clazz.getName());
				}
				definition = type2Definitions.get(type);
				if (definition != null) {
					type2Definitions.put(clazz, definition);
				}
			}
		}
		return definition;
	}

	// public Collection<Class<?>> getSimpleTypes() {
	// return simples.keySet();
	// }
	//
	// /**
	// * 获取枚举定义
	// *
	// * @param index
	// * @return
	// */
	// public Class<?> getSimpleType(int index) {
	// return Class.class.cast(simples.getKey(index));
	// }
	// /**
	// * 获取枚举定义
	// *
	// * @param clazz
	// * @return
	// */
	// public Integer getSimpleCode(Class<?> clazz) {
	// return Integer.class.cast(simples.get(clazz));
	// }

	/**
	 * 检查指定的类型是否符合规范
	 * 
	 * <pre>
	 * 不能有擦拭类型和通配类型
	 * </pre>
	 * 
	 * @param type
	 * @return
	 */
	private static boolean check(Type type) {
		if (type instanceof ParameterizedType) {
			// 泛型类型
			ParameterizedType parameterizedType = (ParameterizedType) type;
			if (!check(parameterizedType.getRawType())) {
				return false;
			}
			for (Type value : parameterizedType.getActualTypeArguments()) {
				if (!check(value)) {
					return false;
				}
			}
		} else if (type instanceof TypeVariable) {
			// 擦拭类型
			return false;
		} else if (type instanceof GenericArrayType) {
			// 数组类型
			GenericArrayType genericArrayType = (GenericArrayType) type;
			if (!check(genericArrayType.getGenericComponentType())) {
				return false;
			}
		} else if (type instanceof WildcardType) {
			// 通配类型
			return false;
		} else if (type instanceof Class) {
			return true;
		}
		return true;
	}

	@Override
	public boolean equals(Object object) {
		if (this == object)
			return true;
		if (object == null)
			return false;
		if (getClass() != object.getClass())
			return false;
		CodecDefinition that = (CodecDefinition) object;
		EqualsBuilder equal = new EqualsBuilder();
		equal.append(this.code2Definitions, that.code2Definitions);
		return equal.isEquals();
	}

	@Override
	public int hashCode() {
		HashCodeBuilder hash = new HashCodeBuilder();
		hash.append(code2Definitions);
		return hash.toHashCode();
	}

	public static CodecDefinition instanceOf(Collection<Type> types) {
		CodecDefinition definition = new CodecDefinition();
		// 遍历与排序所有依赖的类型
		TreeSet<Class<?>> classes = new TreeSet<>(typeComparator);
		for (Type type : types) {
			ClassUtility.findDependentClasses(type, classes);
		}
		classes.addAll(CodecSpecification.type2Specifitions.keySet());
		HashMap<Class<?>, Integer> codes = new HashMap<>();
		for (Class<?> clazz : classes) {
			if (clazz.isArray()) {
				continue;
			}
			int code = definition.codeManager.incrementAndGet();
			codes.put(clazz, code);
		}
		for (Class<?> clazz : classes) {
			if (clazz.isArray()) {
				continue;
			}
			if (definition.type2Definitions.get(clazz) == null) {
				ClassDefinition classDefinition = ClassDefinition.instanceOf(clazz, codes);
				definition.code2Definitions.add(classDefinition);
				definition.type2Definitions.put(clazz, classDefinition);
			}
		}
		return definition;
	}

	public static CodecDefinition fromBytes(byte[] bytes) throws Exception {
		CodecDefinition definition = new CodecDefinition();
		// 解压解密
		byte[] unzip = PressUtility.unzip(bytes, 30, TimeUnit.SECONDS);
		ByteArrayInputStream byteArrayInputStream = new ByteArrayInputStream(unzip);
		DataInputStream dataInputStream = new DataInputStream(byteArrayInputStream);
		// 类型描述
		while (dataInputStream.available() > 0) {
			ClassDefinition classDefinition = ClassDefinition.readFrom(dataInputStream);
			int code = classDefinition.getCode();
			Type clazz = classDefinition.getType();
			definition.code2Definitions.add(classDefinition);
			if (clazz != null) {
				definition.type2Definitions.put(clazz, classDefinition);
			}
		}
		return definition;
	}

	public static byte[] toBytes(CodecDefinition definition) throws Exception {
		// 格式:
		// 类型描述
		ByteArrayOutputStream byteArrayOutputStream = new ByteArrayOutputStream();
		DataOutputStream dataOutputStream = new DataOutputStream(byteArrayOutputStream);
		// 类
		for (ClassDefinition classDefinition : definition.code2Definitions) {
			// if
			// (CodecSpecification.type2Specifitions.containsKey(classDefinition.getType()))
			// {
			// continue;
			// }
			ClassDefinition.writeTo(classDefinition, dataOutputStream);
		}
		// 加密压缩
		byte[] bytes = byteArrayOutputStream.toByteArray();
		byte[] zip = PressUtility.zip(bytes, 5);
		return zip;
	}

}
