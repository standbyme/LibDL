package com.jstarcraft.core.utility;

import java.util.List;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.util.xml.DomUtils;
import org.w3c.dom.Element;

/**
 * XML工具
 * 
 * @author Birdy
 */
public class XmlUtility extends DomUtils {

	private static final Logger logger = LoggerFactory.getLogger(XmlUtility.class);

	/**
	 * 获取元素中指定标签的唯一元素
	 * 
	 * @param element
	 * @param tag
	 * @return
	 */
	public static Element getUniqueElement(Element element, String tag) {
		List<Element> elements = DomUtils.getChildElementsByTagName(element, tag);
		if (elements.size() != 1) {
			String message = StringUtility.format("指定的标签[{}]在元素[{}]中不是唯一", tag, element);
			logger.error(message);
			throw new IllegalArgumentException(message);
		}
		return elements.get(0);
	}

}
