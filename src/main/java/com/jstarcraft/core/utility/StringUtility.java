package com.jstarcraft.core.utility;

import java.nio.charset.Charset;

import org.apache.commons.codec.CharEncoding;
import org.apache.commons.lang3.StringEscapeUtils;
import org.apache.commons.lang3.StringUtils;
import org.apache.commons.lang3.builder.ReflectionToStringBuilder;
import org.slf4j.helpers.FormattingTuple;
import org.slf4j.helpers.MessageFormatter;

/**
 * 字符串工具
 * 
 * @author Birdy
 *
 */
public class StringUtility extends StringUtils {

	/** 字符串编码 */
	public static final Charset CHARSET = Charset.forName(CharEncoding.UTF_8);

	/**
	 * 使用指定参数格式化指定模板,并转换为字符串
	 * 
	 * @param template
	 * @param parameters
	 * @return
	 */
	public static final String format(String template, Object... parameters) {
		FormattingTuple formatter = MessageFormatter.arrayFormat(template, parameters);
		return formatter.getMessage();
	}

	/**
	 * 将指定对象通过反射转换为字符串
	 * 
	 * @param object
	 * @return
	 */
	public static final String reflect(Object object) {
		return ReflectionToStringBuilder.toString(object);
	}

	/**
	 * 对字符串执行CSV转义(TODO 考虑与CvsUtility整合)
	 * 
	 * @param string
	 * @return
	 */
	public static final String escapeCsv(String string) {
		return StringEscapeUtils.escapeCsv(string);
	}

	/**
	 * 对字符串执行CSV翻译(TODO 考虑与CvsUtility整合)
	 * 
	 * @param string
	 * @return
	 */
	public static final String unescapeCsv(String string) {
		return StringEscapeUtils.unescapeCsv(string);
	}

}
