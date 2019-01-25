package com.jstarcraft.core.codec.csv.converter;

import java.lang.reflect.ParameterizedType;
import java.lang.reflect.Type;
import java.util.Collection;
import java.util.Iterator;

import org.apache.commons.csv.CSVPrinter;

import com.jstarcraft.core.codec.csv.CsvReader;
import com.jstarcraft.core.codec.csv.CsvWriter;
import com.jstarcraft.core.codec.specification.ClassDefinition;
import com.jstarcraft.core.codec.specification.CodecSpecification;
import com.jstarcraft.core.utility.StringUtility;
import com.jstarcraft.core.utility.TypeUtility;

/**
 * 集合转换器
 * 
 * @author Birdy
 *
 */
public class CollectionConverter implements CsvConverter<Collection<Object>> {

	@Override
	public Collection<Object> readValueFrom(CsvReader context, Type type) throws Exception {
		// TODO 处理null
		Iterator<String> in = context.getInputStream();
		String check = in.next();
		if (StringUtility.isEmpty(check)) {
			return null;
		}
		int length = Integer.valueOf(check);
		ParameterizedType parameterizedType = ParameterizedType.class.cast(type);
		Type[] types = parameterizedType.getActualTypeArguments();
		Class<?> clazz = TypeUtility.getRawType(type, null);
		ClassDefinition definition = context.getClassDefinition(clazz);
		Collection<Object> collection = (Collection) definition.getInstance();
		Class<?> elementClazz = TypeUtility.getRawType(types[0], null);
		CsvConverter converter = context.getCsvConverter(CodecSpecification.getSpecification(elementClazz));
		for (int index = 0; index < length; index++) {
			Object element = converter.readValueFrom(context, types[0]);
			collection.add(element);
		}
		return collection;
	}

	@Override
	public void writeValueTo(CsvWriter context, Type type, Collection<Object> value) throws Exception {
		// TODO 处理null
		CSVPrinter out = context.getOutputStream();
		if (value == null) {
			out.print(StringUtility.EMPTY);
			return;
		}
		ParameterizedType parameterizedType = ParameterizedType.class.cast(type);
		Type[] types = parameterizedType.getActualTypeArguments();
		Collection<?> collection = Collection.class.cast(value);
		out.print(collection.size());
		Class<?> elementClazz = TypeUtility.getRawType(types[0], null);
		CsvConverter converter = context.getCsvConverter(CodecSpecification.getSpecification(elementClazz));
		for (Object element : collection) {
			converter.writeValueTo(context, types[0], element);
		}
		return;
	}

}
