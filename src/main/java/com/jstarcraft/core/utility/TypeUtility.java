package com.jstarcraft.core.utility;

import java.lang.reflect.Type;
import java.util.List;

import org.apache.commons.lang3.reflect.TypeUtils;

import com.fasterxml.jackson.databind.JavaType;

/**
 * 类型工具
 * 
 * @author Birdy
 *
 */
public class TypeUtility extends TypeUtils {

	/**
	 * 将JavaType转换为Type
	 * 
	 * @param source
	 * @return
	 */
	public static Type convertType(JavaType source) {
		Type target = null;

		if (source.isArrayType()) {
			// 数组类型
			target = source.getRawClass();
		} else if (source.hasGenericTypes()) {
			// 泛型类型
			List<JavaType> types = source.getBindings().getTypeParameters();
			Type[] generics = new Type[types.size()];
			int index = 0;
			for (JavaType type : types) {
				generics[index++] = convertType(type);
			}
			Class<?> clazz = source.getRawClass();
			target = TypeUtility.parameterize(clazz, generics);
		} else {
			target = source.getRawClass();
		}

		return target;
	}

}
