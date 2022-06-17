import React, { ReactElement } from "react";
import { Select } from "antd";
import { LANGUAGES } from "../../config";
import { SpeakerInterface } from "../../interfaces";

export default function LanguageSelect({
  value,
  className,
  onChange,
}: {
  value: SpeakerInterface["language"];
  className: string | null;
  onChange: (lang: SpeakerInterface["language"]) => void;
}): ReactElement {
  return (
    <Select value={value} className={className} onChange={onChange}>
      {LANGUAGES.map((el) => (
        <Select.Option value={el.iso6391} key={el.iso6391}>
          {el.name}
        </Select.Option>
      ))}
    </Select>
  );
}

LanguageSelect.defaultProps = {
  className: null,
};
