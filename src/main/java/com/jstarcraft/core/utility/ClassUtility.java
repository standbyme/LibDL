package com.jstarcraft.core.utility;

import java.io.DataInputStream;
import java.io.InputStream;
import java.lang.reflect.Field;
import java.lang.reflect.GenericArrayType;
import java.lang.reflect.Modifier;
import java.lang.reflect.ParameterizedType;
import java.lang.reflect.Type;
import java.lang.reflect.TypeVariable;
import java.lang.reflect.WildcardType;
import java.util.Collection;
import java.util.HashMap;
import java.util.Map;

import org.apache.commons.lang3.ClassUtils;
import org.springframework.util.ReflectionUtils.FieldCallback;
import org.springframework.util.ReflectionUtils.FieldFilter;

/**
 * 基本数据类型工具
 * 
 * @author Birdy
 *
 */
public class ClassUtility extends ClassUtils {

	private static void findDependentClasses(Class<?> clazz, Collection<Class<?>> classes) {
		ReflectionUtility.doWithFields(clazz, new FieldCallback() {
			@Override
			public void doWith(Field field) throws IllegalArgumentException, IllegalAccessException {
				if (!classes.contains(field.getGenericType())) {
					findDependentClasses(field.getGenericType(), classes);
				}
			}
		}, new FieldFilter() {
			@Override
			public boolean matches(Field field) {
				return !(Modifier.isStatic(field.getModifiers()) || Modifier.isTransient(field.getModifiers()));
			}
		});
		for (Class<?> element : classes.toArray(new Class<?>[] {})) {
			if (!classes.contains(element)) {
				findDependentClasses(element, classes);
			}
		}
	}

	public static void findDependentClasses(Type type, Collection<Class<?>> classes) {
		if (type instanceof Class) {
			Class clazz = (Class) type;
			classes.add(clazz);
			findDependentClasses(clazz, classes);
		} else if (type instanceof GenericArrayType) {
			// 数组类型
			GenericArrayType genericArrayType = (GenericArrayType) type;
			findDependentClasses(genericArrayType.getGenericComponentType(), classes);
		} else if (type instanceof ParameterizedType) {
			// 泛型类型
			ParameterizedType parameterizedType = (ParameterizedType) type;
			findDependentClasses(parameterizedType.getRawType(), classes);
			for (Type value : parameterizedType.getActualTypeArguments()) {
				findDependentClasses(value, classes);
			}
		} else if (type instanceof TypeVariable) {
			// 擦拭类型
			throw new IllegalArgumentException("type is " + type);
		} else if (type instanceof WildcardType) {
			// 通配类型
			throw new IllegalArgumentException("type is " + type);
		}
	}

	public static Map<String, byte[]> classes2Bytes(Class<?>... classes) {
		HashMap<String, byte[]> bytes = new HashMap<>();
		for (Class<?> clazz : classes) {
			String name = clazz.getName();
			String path = name.replace('.', '/') + ".class";
			try (InputStream stream = clazz.getClassLoader().getResourceAsStream(path); DataInputStream buffer = new DataInputStream(stream)) {
				byte[] data = new byte[buffer.available()];
				buffer.readFully(data);
				bytes.put(name, data);
			} catch (Exception exception) {
				throw new IllegalArgumentException(exception);
			}
		}
		return bytes;
	}

	public static Class<?>[] bytes2Classes(Map<String, byte[]> bytes) {
		try (MemoryClassLoader classLoader = new MemoryClassLoader(bytes)) {
			Class<?>[] classes = new Class<?>[bytes.size()];
			int index = 0;
			for (String name : bytes.keySet()) {
				classes[index++] = classLoader.loadClass(name);
			}
			return classes;
		} catch (Exception exception) {
			throw new IllegalArgumentException(exception);
		}
	}

}
