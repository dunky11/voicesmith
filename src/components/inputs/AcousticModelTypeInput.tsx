import React, { ReactElement } from "react";
import { Form, Select, Typography } from "antd";
import HelpIcon from "../help/HelpIcon";
import { Rule } from "antd/lib/form";

export default function AcousticModelTypeInput({
    disabled,
    docsUrl,
    rules,
    name
}: {
    disabled: boolean;
    docsUrl: string | null;
    rules: Rule[] | null;
    name: string;
}): ReactElement {
    return (
        <Form.Item
            label={
                <Typography.Text>
                    Acoustic Model Type
                    {docsUrl && <HelpIcon docsUrl={docsUrl} style={{ marginLeft: 8 }} />}
                </Typography.Text>
            }
            name={name}
            rules={rules}
        >
            <Select disabled={disabled}>
                <Select.Option value="english_only">English Only</Select.Option>
                <Select.Option value="multilingual">
                    Multilingual
                </Select.Option>
            </Select>
        </Form.Item>
    );
}

AcousticModelTypeInput.defaultProps = {
    disabled: false,
    docsUrl: null,
    rules: null,
};
